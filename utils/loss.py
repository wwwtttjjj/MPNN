import torch
from torch.autograd import Function
from torch import Tensor
from scipy.spatial.distance import directed_hausdorff
from torch import nn
import numpy as np

class BCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.bce_loss(output, target)
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def iou(outputs: np.array, labels: np.array):

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean()


#######################
import math
import torch
import torch.nn as nn


def evaluate_jaccard(outputs, targets):
    eps = 1e-15
    jaccard_targets = (targets == 1).float()
    jaccard_outputs = torch.sigmoid(outputs)
    # jaccard_outputs =outputs

    intersection = (jaccard_outputs * jaccard_targets).sum()
    union = jaccard_outputs.sum() + jaccard_targets.sum()

    jaccard = (intersection + eps) / (union - intersection + eps)

    return jaccard


def cal_jaccard(dice):
    return dice / 2-dice

def evaluate_dice(jaccard):
    return 2 * jaccard / (1 + jaccard)


class SoftJaccardBCEWithLogitsLoss:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """
    eps = 10e-5

    def __init__(self, jaccard_weight=0.0):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.jacc_weight = jaccard_weight

    def __call__(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)
        jaccard = evaluate_jaccard(outputs, targets)
        log_jaccard = math.log(jaccard + self.eps)
        loss = bce_loss - self.jacc_weight * log_jaccard

        return loss




def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    target = (target == 1).float()
    pres = torch.sigmoid(preds)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = pres.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])

    return res
###################################
def sigmoid_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.sigmoid(input_logits)
    target_softmax = torch.sigmoid(target_logits)
    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

