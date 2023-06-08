from torch.autograd import Function
import torch
import torch.nn as nn


class CustomReLU(Function):

    @staticmethod
    def forward(ctx, input, p):
        mask = input < 0
        unmask = input > 0
        mask = torch.bernoulli(mask * p)
        output = mask * input + unmask * input
        ctx.save_for_backward(input, mask, unmask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, mask, unmask = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input *= (mask + unmask)

        return grad_output


class C_ReLU(nn.Module):

    def __init__(self):
        super(C_ReLU, self).__init__()
        self.w = nn.Parameter(torch.randn(1), requires_grad=True)


    def forward(self, input):
        p = torch.clamp(self.w, 0, 1)
        return CustomReLU.apply(input, p)

