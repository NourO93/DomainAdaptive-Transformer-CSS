# models.py
import torch
import torch.nn as nn
from rtdl import FTTransformer


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradReverseLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradReverse.apply(x, self.lambda_)


class FTDANN(nn.Module):
    """
    FT-Transformer + Domain-Adversarial training.
    """

    def __init__(self, input_dim: int, lambda_grl: float = 0.5):
        super().__init__()
        self.feature_extractor = FTTransformer.make_default(
            n_num_features=input_dim,
            cat_cardinalities=None,
            d_out=64,
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.grl = GradReverseLayer(lambda_=lambda_grl)

        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, D]
        features = self.feature_extractor(x, None)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output
