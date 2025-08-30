from dataclasses import dataclass

import esm
import torch
import torch.nn.functional as F

from model import ESM, ESMConfig


@dataclass
class DistilConfig(ESMConfig):
    teacher_model_name: str = "esm2_t12_35M_UR50D"
    temperature: float = 2.0           
    alpha: float = 0.7 # KL div on masked tokens              


class DistilESM(ESM):
    def __init__(self, config):
        super().__init__(config)
        self.temp = config.temperature
        self.alpha = config.alpha
        self.teacher, _ = getattr(esm.pretrained, config.teacher_model_name)()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def forward(self, toks, mask=None, targets=None):
        student_logits, ce_loss = super().forward(toks, mask, targets)

        if self.training and targets is not None:
            # TRAINING
            with torch.no_grad():
                teacher_output = self.teacher(toks, repr_layers=[], return_contacts=False)
                teacher_logits = teacher_output["logits"] # (B,T,C), float32

            # only masked tokens
            masked_tok_mask = (targets != -100)
            if masked_tok_mask.any():
                kd_loss = F.kl_div(
                    input=F.log_softmax(student_logits[masked_tok_mask] / self.temp, dim=-1),
                    target=F.softmax(teacher_logits[masked_tok_mask] / self.temp, dim=-1),
                    reduction="batchmean"
                ) * self.temp**2
            else:
                kd_loss = 0.0

            total_loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
            return student_logits, total_loss, ce_loss
        else:
            # VALIDATION/INFERENCE
            return student_logits, ce_loss, ce_loss

