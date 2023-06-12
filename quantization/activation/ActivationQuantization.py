
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd, autocast
import math

class ClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, clip_val, learned=False):
        super(ClippedLinearQuantization, self).__init__()

        print("Creating a ClippedLinearQuantization module with  n_bits: {}, clip_val: {}, learned_clip_val: {}".format(
            num_bits, clip_val, learned))
        self.num_bits = num_bits
        self.learned = learned
        if learned:
            self.clip_val = nn.Parameter(torch.tensor([clip_val], device="cuda", requires_grad=True))
        else:
            self.clip_val = clip_val
            self.scale = asymmetric_linear_quantization_params(num_bits, 0, clip_val)
        # self.clip_val = self.clip_val.to(torch.float16)
        self.zero_point = 0

    @autocast()
    def forward(self, input):
        if self.learned:
            with autocast(enabled=False):
                input_clamp = torch.where(input < self.clip_val, input.float(), self.clip_val)
            with torch.no_grad():
                self.scale = asymmetric_linear_quantization_params(self.num_bits, 0, self.clip_val)
        else:
            input_clamp = torch.clamp(input, 0, self.clip_val)

        out = LinearQuantizeSTE.apply(input_clamp, self.scale, self.zero_point)
        return out


def asymmetric_linear_quantization_params(num_bits, sat_min, sat_max):
    if sat_min > sat_max:
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1
    diff = sat_max - sat_min
    scale = n / diff
    return scale


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, scale, zero_point):
        # quantize and dequantize
        output = (torch.round(scale * input - zero_point) + zero_point) / scale

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None


class EWGS_discretizer(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, x_in, num_levels, scaling_factor):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)

        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x_in - x_out)
        return x_out

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(g) * diff
        return g * scale, None, None


class EWGSActivationQuantizer(nn.Module):
    def __init__(self, num_bits, bkwd_scaling_factorA=0.001):
        super(EWGSActivationQuantizer, self).__init__()

        print("Creating a EWGSActivationQuantizer module with  n_bits: {}, bkwd_scaling_factor: {}".format(num_bits,
                                                                                                           bkwd_scaling_factorA))

        self.act_levels = 2 ** num_bits
        self.uA = nn.Parameter(data=torch.tensor(0).float())
        self.lA = nn.Parameter(data=torch.tensor(0).float())
        self.register_buffer('bkwd_scaling_factorA', torch.tensor(bkwd_scaling_factorA).float())

        self.output_scale = nn.Parameter(data=torch.tensor(1).float())
        self.initialized = False

        self.hook_Qvalues = False
        self.buff_act = None

    @autocast()
    def forward(self, input):

        if not self.initialized:
            with torch.no_grad():
                self.uA.data.fill_(input.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
                self.lA.data.fill_(input.min())

        x = (input - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1)  # [0, 1]
        x = EWGS_discretizer.apply(x, self.act_levels, self.bkwd_scaling_factorA)
        if self.hook_Qvalues:
            self.buff_act = x
            self.buff_act.retain_grad()
        return x