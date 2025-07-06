import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = ['GradientReversalLayer']


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(alpha_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        alpha_, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


class GradientReversalLayer(nn.Module):

    def __init__(self, warm_up=4000):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()
        self.n_iters = 0
        self.warm_up = warm_up

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        self.n_iters += 1
        alpha = min(self.n_iters / self.warm_up, 1)
        alpha = torch.tensor(alpha, requires_grad=False)
        return revgrad(input_, alpha)
