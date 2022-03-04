from math import pi, sqrt
import torch.nn.functional as F

import torch
from torch.nn import Module, BCEWithLogitsLoss, CrossEntropyLoss,Softmax, MSELoss,NLLLoss


class VFAELoss(Module):
    """
    Loss function for training the Variational Fair Auto Encoder.
    """

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha

        self.bce = BCEWithLogitsLoss()
        self.ce = CrossEntropyLoss()
        self.ce_logits = CrossEntropyLoss(reduction='none')
        self.mse = MSELoss()
        self.nll = NLLLoss()

    def forward(self, y_pred, x,y,s):
        y = y.squeeze().long()

        zeros = torch.zeros_like(y_pred['z1_enc_logvar'])
        kl_loss = self.kl_gaussian(y_pred['z_enc_logvar'],
                                    y_pred['z_enc_mu'],
                                    zeros,
                                    zeros)


  
        reconstruction_loss = self.mse(y_pred['x_decoded'], x) + self.alpha*self.bce(y_pred['s_decoded'],s.unsqueeze(-1).float())

        loss = reconstruction_loss +kl_loss

        return loss


    def kl_gaussian(logvar_a, mu_a, logvar_b, mu_b):
        """
        average kl betwen two gaussians
        """
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b).square()) / logvar_b.exp()
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl.mean()


