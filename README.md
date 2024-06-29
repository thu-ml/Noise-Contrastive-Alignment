NCCL_P2P_DISABLE=1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port=7000 run_reward.py yamls/reward_qlora.yaml --gradient_accumulation_steps=4 --beta=0.01 --loss_type=NCA --output_dir=data/test_run


NCCL_P2P_DISABLE=1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port=7000 run_preference.py yamls/preference_qlora.yaml --gradient_accumulation_steps=4 --beta=0.01 --loss_type=NCA --output_dir=data/test_run


[Released Models](https://huggingface.co/collections/ChenDRAG/noise-contrastive-alignment-model-collection-65c49b9cb25522fdb035a206)
