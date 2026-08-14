import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    """P-Loss defined in the FreSH paper (Eq. 13).

    L_P = - (1/N) sum_i log(y_hat_i) + lambda_P * (1/N) sum_i (1 - y_hat_i)^2,
    where y_hat_i is the predicted probability of the true class for sample i.
    """

    def __init__(self, lambda_p=1.0, reduction='mean'):
        super(PolyLoss, self).__init__()
        self.lambda_p = lambda_p
        self.reduction = reduction

    def forward(self, input, target):
        # input: [B, num_classes] logits; target: [B] integer labels.
        log_probs = F.log_softmax(input, dim=1)
        probs = log_probs.exp()

        true_log_probs = log_probs.gather(1, target.reshape(-1, 1)).squeeze(1)
        true_probs = probs.gather(1, target.reshape(-1, 1)).squeeze(1)

        ce = -true_log_probs
        second_order = (1.0 - true_probs) ** 2
        loss = ce + self.lambda_p * second_order

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
