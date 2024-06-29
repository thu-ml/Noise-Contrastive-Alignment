#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import logging
import sys

import torch
import transformers
transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "zephyr_checkpoint"
transformers.trainer.PREFIX_CHECKPOINT_DIR = "zephyr_checkpoint"
from transformers import AutoModelForCausalLM, set_seed
from accelerate import Accelerator
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from peft import PeftConfig, PeftModel
from preference_trainer import NCATrainer_pairwise
from configs import NCAConfig
from data_utils import apply_chat_template

logger = logging.getLogger(__name__)

from huggingface_hub.utils._validators import HFValidationError 
from huggingface_hub import list_repo_files

def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool: 
    try: # Try first if model on a Hub repo 
        repo_files = list_repo_files(model_name_or_path, revision=revision) 
    except HFValidationError: # If not, check local repo 
        import os
        repo_files = os.listdir(model_name_or_path) 
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, NCAConfig))
    model_args, data_args, training_args = parser.parse()
    training_args.logging_dir= os.path.join(training_args.output_dir,"manual_runs")
    training_args.run_name = os.path.basename(training_args.output_dir)

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
        load_from_cache_file=False
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map(),
        quantization_config=get_quantization_config(model_args),
    )
    print("Using {}######################################################".format(model_args.use_flash_attention_2))

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        print("merging base model")
        # load the model, merge the adapter weights and unload the adapter
        # Note: to run QLora, you will need to merge the based model separately as the merged model in 16bit
        logger.info(f"Merging peft adapters for {model_args.model_name_or_path=}")

        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)

        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model, model_args.model_name_or_path, revision=model_args.model_revision
        )
        model.eval()
        model = model.merge_and_unload()
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    dpo_trainer = NCATrainer_pairwise(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        loss_type=training_args.loss_type,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
    )
    print("#############################################     Trainer init finished")
    ###############
    # Training loop
    ###############
    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    logger.info("*** Training complete ***")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = dpo_trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["test"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["test"]))
        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    dpo_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        dpo_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        dpo_trainer.model.config.use_cache = True
        dpo_trainer.model.config.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub is True:
            dpo_trainer.push_to_hub()

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
