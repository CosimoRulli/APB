import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from ..activation.ActivationQuantization import *
from ..activation.ActivationBinarization import *
import numpy as np
import torch.nn.init as init


class FixedPruneQuantizeBinarizationSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, sparsity, mask, inplace=False,):
        if inplace:
            ctx.mark_dirty(input)

        if mask is None:
            mask = torch.gt(m, torch.abs((input.abs() - alpha)/delta))

        ctx.save_for_backward(input, alpha, delta, m, mask)

        input = torch.where(mask, torch.sign(input) * alpha, input)
        #Ensure no zero is present
        input = torch.where(input == 0, torch.ones_like(input) * alpha, input)
        return input


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        #We use g(x) = Identity(x) in this implementation, hence dB/dx = 1
        inp, alpha, delta, m, mask = ctx.saved_tensors
        w_hat = torch.abs((inp.abs() - alpha) / delta)
        if mask is None:
            mask = torch.gt(m, w_hat)

        filter_grad = torch.where(mask,  grad_output, torch.zeros(grad_output.shape).cuda())

        #grad_delta = torch.mean(filter_grad * torch.abs((alpha - inp.abs()))) * 1 / delta
        grad_delta = torch.mean(filter_grad * torch.sign(inp) * (alpha - inp.abs())) * 1 / delta
        #grad_alpha = - torch.mean(filter_grad * torch.sign(w_hat))
        grad_alpha = - torch.mean(filter_grad * torch.sign(inp))

        return grad_output, grad_alpha.view(-1), grad_delta.view(-1), None, None, None, None




class FixedPruneQuantizerConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 act_quantization_mode=None):

        super(FixedPruneQuantizerConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sparsity = act_quantization_mode
        self.apb = None


    def forward(self, input):

        #Weight Quantization
        self.quant_param = self.apb.apply(self.weight, self.alpha, self.delta, self.m, self.frozen_mask, self.min_alpha)


        activation = input

        output = F.conv2d(activation, self.quant_param, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)


        return output

    def init_alpha_delta(self):

        self.alpha = nn.Parameter(torch.tensor([torch.mean(self.weight.abs())], device="cuda", requires_grad=True))
        self.register_parameter("alpha", self.alpha)
        #self.delta = nn.Parameter(torch.tensor([3 * torch.std(self.weight.abs())], device="cuda", requires_grad=True))
        self.delta = nn.Parameter(torch.tensor([3 * torch.std(self.weight)], device="cuda", requires_grad=True))
        self.register_parameter("delta", self.delta)
        print("Initializing alpha: {:.3f} ".format(self.alpha.item()))
        print("Initializing delta: {:.3f} ".format(self.delta.item()))

    def freeze_mask(self):
        w_hat = torch.abs((self.weight.abs() - self.alpha) / self.delta)
        self.frozen_mask = torch.gt(self.m, w_hat).to(torch.uint8)
        #self.frozen_mask.requires_grad = False

    def count_unique_params(self):
        print(len(torch.unique(self.quant_param)))

    def reset_bin_params(self):
        with torch.no_grad():
            w_hat = torch.abs((self.weight.abs() - self.alpha) / self.delta)
            temp_mask = torch.gt(self.m, w_hat)
            re_init_weight = init.kaiming_normal_(torch.zeros_like(self.weight, requires_grad=True, device=self.weight.device))
            self.weight[temp_mask] = re_init_weight[temp_mask]

class AutomaticPruneQuantizeBinarizationSTE_absval(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, alpha, delta, m, mask=None,  min_alpha=None, inplace=False,):
        if inplace:
            ctx.mark_dirty(input)
        if min_alpha:
            alpha = torch.max(alpha, min_alpha)

        if mask is None:
            mask = torch.gt(m, torch.abs((input.abs() - alpha)/delta))

        ctx.save_for_backward(input, alpha, delta, m, mask)

        input = torch.where(mask, torch.sign(input) * torch.mean(torch.abs(input[mask])), input)
        #Ensure no zero is present
        input = torch.where(input == 0, torch.ones_like(input) * alpha, input)
        return input


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        #We use g(x) = Identity(x) in this implementation, hence dB/dx = 1
        inp, alpha, delta, m, mask = ctx.saved_tensors
        w_hat = torch.abs((inp.abs() - alpha) / delta)
        if mask is None:
            mask = torch.gt(m, w_hat)

        filter_grad = torch.where(mask,  grad_output, torch.zeros(grad_output.shape).cuda())

        #grad_delta = torch.mean(filter_grad * torch.abs((alpha - inp.abs()))) * 1 / delta
        grad_delta = torch.mean(filter_grad * torch.sign(inp) * (alpha - inp.abs())) * 1 / delta
        #grad_alpha = - torch.mean(filter_grad * torch.sign(w_hat))
        grad_alpha = - torch.mean(filter_grad * torch.sign(inp))

        return grad_output, grad_alpha.view(-1), grad_delta.view(-1), None, None, None, None





class AutomaticPruneQuantizeBinarizationSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, alpha, delta, m, mask=None,  min_alpha=None, inplace=False,):
        if inplace:
            ctx.mark_dirty(input)
        if min_alpha:
            alpha = torch.max(alpha, min_alpha)

        if mask is None:
            mask = torch.gt(m, torch.abs((input.abs() - alpha)/delta))

        ctx.save_for_backward(input, alpha, delta, m, mask)

        input = torch.where(mask, torch.sign(input) * alpha, input)
        #Ensure no zero is present
        input = torch.where(input == 0, torch.ones_like(input) * alpha, input)
        return input


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        #We use g(x) = Identity(x) in this implementation, hence dB/dx = 1
        inp, alpha, delta, m, mask = ctx.saved_tensors
        w_hat = torch.abs((inp.abs() - alpha) / delta)
        if mask is None:
            mask = torch.gt(m, w_hat)

        filter_grad = torch.where(mask,  grad_output, torch.zeros(grad_output.shape).cuda())

        #grad_delta = torch.mean(filter_grad * torch.abs((alpha - inp.abs()))) * 1 / delta
        grad_delta = torch.mean(filter_grad * torch.sign(inp) * (alpha - inp.abs())) * 1 / delta
        #grad_alpha = - torch.mean(filter_grad * torch.sign(w_hat))
        grad_alpha = - torch.mean(filter_grad * torch.sign(inp))

        return grad_output, grad_alpha.view(-1), grad_delta.view(-1), None, None, None, None



class AutomaticPruneQuantizeBinarizationSTESum(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, alpha, delta, m, mask=None,  min_alpha=None, inplace=False,):
        if inplace:
            ctx.mark_dirty(input)
        if min_alpha:
            alpha = torch.max(alpha, min_alpha)

        if mask is None:
            mask = torch.gt(m, torch.abs((input.abs() - alpha)/delta))

        ctx.save_for_backward(input, alpha, delta, m, mask)

        input = torch.where(mask, torch.sign(input) * alpha, input)
        #Ensure no zero is present
        input = torch.where(input == 0, torch.ones_like(input) * alpha, input)
        return input


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        #We use g(x) = Identity(x) in this implementation, hence dB/dx = 1
        inp, alpha, delta, m, mask = ctx.saved_tensors
        w_hat = torch.abs((inp.abs() - alpha) / delta)
        if mask is None:
            mask = torch.gt(m, w_hat)

        filter_grad = torch.where(mask,  grad_output, torch.zeros(grad_output.shape).cuda())

        #grad_delta = torch.mean(filter_grad * torch.abs((alpha - inp.abs()))) * 1 / delta
        grad_delta = torch.sum(filter_grad * torch.sign(inp) * (alpha - inp.abs())) * 1 / delta
        #grad_alpha = - torch.mean(filter_grad * torch.sign(w_hat))
        grad_alpha = - torch.sum(filter_grad * torch.sign(inp))

        return grad_output, grad_alpha.view(-1), grad_delta.view(-1), None, None, None, None




class APQConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
             act_quantization_mode=None):

        super(APQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.beta = None
        self.activation_quantizer = None

        if act_quantization_mode is not None:
            if act_quantization_mode.startswith("dorefa"):
                # dorefa-n_bits-clip_val-learned
                learned_clip_val = False
                if "learned" in act_quantization_mode:
                    learned_clip_val = True
                n_bits = int(act_quantization_mode.split("-")[1])
                clip_val = int(act_quantization_mode.split("-")[2])
                self.activation_quantizer = ClippedLinearQuantization(num_bits=n_bits, clip_val=clip_val, learned=learned_clip_val)
            if act_quantization_mode.startswith("ewgs"):
                n_bits = int(act_quantization_mode.split("-")[1])
                self.activation_quantizer = EWGSActivationQuantizer(num_bits=n_bits)
            if act_quantization_mode == "binary-recu":
                self.activation_quantizer = ReCuActivationQuantization()
                #self.beta = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
            if act_quantization_mode == "binary-id":
                self.activation_quantizer = IdentityActivationBinarization()
            assert self.activation_quantizer
        self.register_buffer("quant_param", torch.zeros(self.weight.shape))
        m = 1
        self.register_buffer("m", torch.tensor([m]))
        self.register_buffer("frozen_mask", None)
        #self.frozen_mask = None
        #self.register_parameter("frozen_mask", nn.Parameter(torch.zeros_like(self.weight, dtype=torch.uint8), requires_grad=False))
        #self.register_buffer("frozen_mask", torch.zeros_like(self.weight, dtype=torch.uint8))
        #self.frozen_mask = nn.Parameter(None, device="cuda", requires_grad=False)
        #self.register_parameter("frozen_mask", self.frozen_mask)

        self.min_alpha = None
        self.clustering = False
        self.normalize_param = False
        self.apb = AutomaticPruneQuantizeBinarizationSTE


    def forward(self, input):

        #Weight Quantization
        self.quant_param = self.apb.apply(self.weight, self.alpha, self.delta, self.m, self.frozen_mask, self.min_alpha)

        #Activation Quantization
        if self.activation_quantizer:
            activation = self.activation_quantizer(input)
        else:
            activation = input

        output = F.conv2d(activation, self.quant_param, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        if isinstance(self.activation_quantizer, EWGSActivationQuantizer):
            if not self.activation_quantizer.initialized:
                output_no_quant = F.conv2d(input, self.quant_param, self.bias,
                              self.stride, self.padding,
                              self.dilation, self.groups)
                self.activation_quantizer.output_scale.data.fill_(output_no_quant.abs().mean() / output.abs().mean())
                self.activation_quantizer.initialized = True

            output *= torch.abs(self.activation_quantizer.output_scale)

        if self.beta is not None:
            output *= self.beta

        return output

    def init_alpha_delta(self):

        self.alpha = nn.Parameter(torch.tensor([torch.mean(self.weight.abs())], device="cuda", requires_grad=True))
        self.register_parameter("alpha", self.alpha)
        #self.delta = nn.Parameter(torch.tensor([3 * torch.std(self.weight.abs())], device="cuda", requires_grad=True))
        self.delta = nn.Parameter(torch.tensor([3 * torch.std(self.weight)], device="cuda", requires_grad=True))
        self.register_parameter("delta", self.delta)
        print("Initializing alpha: {:.3f} ".format(self.alpha.item()))
        print("Initializing delta: {:.3f} ".format(self.delta.item()))

    def freeze_mask(self):
        w_hat = torch.abs((self.weight.abs() - self.alpha) / self.delta)
        self.frozen_mask = torch.gt(self.m, w_hat).to(torch.uint8)
        #self.frozen_mask.requires_grad = False

    def count_unique_params(self):
        print(len(torch.unique(self.quant_param)))

    def reset_bin_params(self):
        with torch.no_grad():
            w_hat = torch.abs((self.weight.abs() - self.alpha) / self.delta)
            temp_mask = torch.gt(self.m, w_hat)
            re_init_weight = init.kaiming_normal_(torch.zeros_like(self.weight, requires_grad=True, device=self.weight.device))
            self.weight[temp_mask] = re_init_weight[temp_mask]


# class APBChannelWise(APQConv2d):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                  act_quantization_mode=None):
#
#         super(APBChannelWise, self).__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                                              act_quantization_mode=None)
#
#         self.apb = AutomaticPruneQuantizeBinarizationSTECW


    # def init_alpha_delta(self):
    #
    #     self.alpha = nn.Parameter(torch.tensor(torch.mean(self.weight.abs(), dim=[1,2,3], keepdim=True), device="cuda", requires_grad=True))
    #     self.register_parameter("alpha", self.alpha)
    #     #self.delta = nn.Parameter(torch.tensor([3 * torch.std(self.weight.abs())], device="cuda", requires_grad=True))
    #     self.delta = nn.Parameter(torch.tensor(3 * torch.std(self.weight, dim=[1,2,3], keepdim=True), device="cuda", requires_grad=True))
    #     self.register_parameter("delta", self.delta)
    #     print("Initializing alpha: ", self.alpha.shape)
    #     print("Initializing delta: ", self.delta.shape)


# class AutomaticPruneQuantizeBinarizationSTECW(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd
#     def forward(ctx, input, alpha, delta, m, mask=None,  min_alpha=None, inplace=False,):
#         if inplace:
#             ctx.mark_dirty(input)
#         if min_alpha:
#             alpha = torch.max(alpha, min_alpha)
#
#         if mask is None:
#             mask = torch.gt(m, torch.abs((input.abs() - alpha)/delta))
#
#         ctx.save_for_backward(input, alpha, delta, m, mask)
#
#         input = torch.where(mask, torch.sign(input) * alpha, input)
#         #Ensure no zero is present
#         input = torch.where(input == 0, torch.ones_like(input) * alpha, input)
#         return input


    # @staticmethod
    # @custom_bwd
    # def backward(ctx, grad_output):
    #     #We use g(x) = Identity(x) in this implementation, hence dB/dx = 1
    #     inp, alpha, delta, m, mask = ctx.saved_tensors
    #     w_hat = torch.abs((inp.abs() - alpha) / delta)
    #     if mask is None:
    #         mask = torch.gt(m, w_hat)
    #
    #     filter_grad = torch.where(mask,  grad_output, torch.zeros(grad_output.shape).cuda())
    #
    #     #grad_delta = torch.mean(filter_grad * torch.abs((alpha - inp.abs()))) * 1 / delta
    #     grad_delta = torch.mean(filter_grad * torch.sign(inp) * (alpha - inp.abs()), dim=[1, 2, 3], keepdim=True) * 1 / delta
    #     #grad_alpha = - torch.mean(filter_grad * torch.sign(w_hat))
    #     grad_alpha = - torch.mean(filter_grad * torch.sign(inp), dim=[1, 2, 3], keepdim=True)
    #
    #     return grad_output, grad_alpha, grad_delta, None, None, None, None
