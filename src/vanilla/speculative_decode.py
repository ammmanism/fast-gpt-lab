"""
Speculative Decoding Engine — fast-gpt-lab
Accelerates autoregressive inference by using a small "Draft" model 
to predict K tokens, verified in parallel by the "Target" model.
"""

import torch
import torch.nn as nn


class SpeculativeDecoder:
    def __init__(self, target_model: nn.Module, draft_model: nn.Module, gamma: int = 4):
        """
        gamma: Number of tokens the draft model predicts before verification.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma
        self.draft_model.eval()
        self.target_model.eval()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Performs the speculative decoding loop.
        Achieves ~2x-3x speedup on memory-bound generation tasks.
        """
        n = 0
        while n < max_new_tokens:
            # 1. Draft phase: generate gamma tokens autoregressively with the small model
            draft_ids = input_ids.clone()
            for _ in range(self.gamma):
                logits, _ = self.draft_model(draft_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
                draft_ids = torch.cat((draft_ids, next_token), dim=1)

            # 2. Verification phase: single forward pass with the large target model
            # We verify the draft tokens all at once in parallel
            target_logits, _ = self.target_model(draft_ids)

            # 3. Acceptance logic
            n_accepted = 0
            for i in range(self.gamma):
                # Compare the argmax of the target model with what the draft model proposed
                target_token = torch.argmax(target_logits[:, -(self.gamma + 1) + i, :], dim=-1)
                draft_token = draft_ids[:, -(self.gamma) + i]

                if target_token == draft_token:
                    n_accepted += 1
                else:
                    # Reject the rest if one is wrong
                    break

            # Keep accepted tokens and add the next correct token from the target model
            valid_length = input_ids.shape[1] + n_accepted
            input_ids = draft_ids[:, :valid_length]

            # Add the final verified token
            final_token = torch.argmax(target_logits[:, valid_length - 1, :], dim=-1).unsqueeze(1)
            input_ids = torch.cat((input_ids, final_token), dim=1)

            n += (n_accepted + 1)

        return input_ids[:, :input_ids.shape[1] + max_new_tokens]
