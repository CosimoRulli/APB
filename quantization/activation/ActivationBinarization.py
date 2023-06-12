import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd, autocast
import math


class ReCuActivationQuantization(nn.Module):
    def __init__(self):
        super(ReCuActivationQuantization, self).__init__()

    def forward(self, input):

        if self.training:
            a = input / torch.sqrt(input.var([1, 2, 3], keepdim=True) + 1e-5)
        else:
            a = input
        return ReCuBinarizationSTE.apply(a)

class IdentityActivationBinarization(nn.Module):
    def __init__(self):
        super(IdentityActivationBinarization, self).__init__()

    def forward(self, input):

        a = input
        return IdentityBinarizationSTE.apply(a)




class ReCuBinarizationSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        #out = torch.where(out == 0, torch.ones_like(out), out)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class IdentityBinarizationSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        out = torch.sign(input)
        out = torch.where(out == 0, torch.ones_like(out), out)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # input = ctx.saved_tensors[0]

        return grad_output


class ExtendedApproxSignSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input):

        l = torch.min(input)
        u = torch.max(input)
        ctx.save_for_backward(input, l, u)
        out = torch.sign(input)
        out = torch.where(out == 0, torch.ones_like(out), out)

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # input = ctx.saved_tensors[0]
        input, l, u = ctx.saved_tensors

        grad_input = torch.zeros_like(input)

        grad_input = torch.where((l <= input) & (input < 0), 2 - (2/l) * input, grad_input)
        grad_input = torch.where((input >= 0) & (u > input), 2 - (2/u) * input, grad_input)


        #l_mask = (l < input) & (input < 0)
        #grad_input[l_mask] = 2 - (2/l) * input

        #u_mask = (input >= 0) & (u > input)
        #grad_input[u_mask] = 2 - (2/u) * input

        grad_input = grad_input * grad_output.clone()
        return grad_input




class OurBinarizationActivation(nn.Module):
    def __init__(self, ste=ReCuBinarizationSTE):
        super(OurBinarizationActivation, self).__init__()

    def forward(self, input):
        a = input / torch.sqrt(input.var([1, 2, 3], keepdim=True) + 1e-5)
        a = ExtendedApproxSignSTE.apply(a)
        return a

class MeanNormalization(nn.Module):
    def __init__(self, ste=ReCuBinarizationSTE):
        super(MeanNormalization, self).__init__()
        self.ste = ste

    def forward(self, input):
        a = input - input.mean([1, 2, 3], keepdim=True)
        return self.ste.apply(a)


class MeanVarNormalization(nn.Module):
    def __init__(self, ste=ReCuBinarizationSTE):
        super(MeanVarNormalization, self).__init__()
        self.ste = ste

    def forward(self, input):
        a = input - input.mean([1, 2, 3], keepdim=True)
        a = a / torch.sqrt(input.var([1, 2, 3], keepdim=True) + 1e-5)
        return self.ste.apply(a)


class NoNormalization(nn.Module):
    def __init__(self, ste=ReCuBinarizationSTE):
        super(NoNormalization, self).__init__()
        self.ste = ste

    def forward(self, input):
        return self.ste.apply(input)


class VarNormalization(nn.Module):
    def __init__(self, ste=ReCuBinarizationSTE):
        super(VarNormalization, self).__init__()
        self.ste = ste
        print(ste)

    def forward(self, input):
        a = input / torch.sqrt(input.var([1, 2, 3], keepdim=True) + 1e-5)
        #a = input
        return self.ste.apply(a)


class IrNetActivationQuantization(nn.Module):
    def __init__(self):
        super(IrNetActivationQuantization, self).__init__()
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        a = input - input.mean([1, 2, 3], keepdim=True)
        a = a / torch.sqrt(input.var([1, 2, 3], keepdim=True) + 1e-5)
        return IrNetActivationSTE.apply(a, self.k, self.t)


class IrNetActivationSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


