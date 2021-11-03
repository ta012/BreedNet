#!https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import time

conv2d_parms = [ 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']


def parse_net(model, input_size, batch_size=-1, device =None, dtypes=None):
    """
    Parse input network to get architecture information
    """
    summary = summary_string(model, input_size, batch_size, device, dtypes)

    return summary

    # return params_info,conv_inpout


def summary_string(model, input_size, batch_size=-1, device=None, dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            # ['module','input_shape','output_shape','kernel_size']
            info_dict = dict()
            info_dict['module'] = str(module.__class__.__name__)

            m_key = "%s-%i" % (class_name, module_idx + 1)

            info_dict["input_shape"] = list(input[0].size())

            info_dict["output_shape"] = list(output.size())

            if str(module.__class__.__name__) == 'Conv2d':
                for j in conv2d_parms:
                    # print(j)
                    info_dict[j] = eval('module.'+j)
            # print(module.__class__.__name__)
            
            if module.__class__.__name__ in dir(torch.nn):
                summary.append(info_dict)
            

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]


    # create properties
    # summary = OrderedDict()
    summary = []

    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)


    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    conv_inpout = []

    return summary

    
if __name__ == "__main__":
    import torchvision
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    n3 = torchvision.models.resnet18().to(DEVICE)
    _= parse_net(n3, (3, 320, 320),device=DEVICE)
    print()