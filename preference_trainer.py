# Code modified from 
import torch
import torch.nn.functional as F
from trl import DPOTrainer
from typing import Tuple
from typing import Literal

class NCATrainer_pairwise(DPOTrainer):
    r"""
    Implementation of the NCA algorithm in pairwise preference settings.
    """

    def __init__(self, *args, loss_type: Literal["InfoNCA", "NCA"] = "InfoNCA", **kwargs):
        super().__init__(*args, loss_type=loss_type, **kwargs)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Implementation of the InfoNCA/NCA loss in pairwise prefernece settings.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        if self.loss_type == "DPO" or self.loss_type == "InfoNCA":
            # Pairwise InfoNCA is Equivalent to DPO loss.
            losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        elif self.loss_type == "NCA":
            # Pairwise NCA differs from DPO by only one line of code.
            losses = - F.logsigmoid(chosen_rewards) - 0.5 * F.logsigmoid(-chosen_rewards) - 0.5 * F.logsigmoid(-rejected_rewards)
        elif self.loss_type == "biasedNCA":
            # Might further prevent decreasing chosen logp
            losses = - F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['DPO', 'InfoNCA', 'NCA', 'biasedNCA']")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()
