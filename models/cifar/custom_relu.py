from torch.autograd import Function
import torch
import torch.nn as nn

class CustomReLU(Function):

    @staticmethod
    def forward(ctx, input):
        p = 0.0
        mask = input <0
        unmask = input>0
        mask = torch.bernoulli(mask*p)
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


    def forward(self, input):
        return CustomReLU.apply(input)

