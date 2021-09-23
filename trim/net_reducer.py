#!https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info,conv_inpout = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)
    print(conv_inpout)

    return params_info,conv_inpout


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

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
    summary = OrderedDict()
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
    for layer in summary:
        print("layer",layer,summary[layer]['input_shape'][1],summary[layer]['output_shape'][1])
        if 'Conv2d' in layer:
          print("yes")
          conv_inpout.append(('Conv2d',summary[layer]['input_shape'][1],summary[layer]['output_shape'][1]))
        elif 'BatchNorm2d-' in layer:
          print("yes")
          conv_inpout.append(('BatchNorm2d',summary[layer]['input_shape'][1],summary[layer]['output_shape'][1]))
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params),conv_inpout

#######################
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, inp : int, oup : int, stride : int , expand_ratio : float):
        super(InvertedResidual, self).__init__()
        # ReLU = nn.ReLU 
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
            )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
            )

        # self.skip_add = nn.quantized.FloatFunctional()
 

    def forward(self, x : torch.Tensor):
        return self.conv(x)
        # if self.use_res_connect:
        #     return self.skip_add.add(x, self.conv(x))
        # else:
        #     return self.conv(x)
#######################
def replace_layers(model, old, new):
    c = 0  
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            # print(module)

            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            # print(module)
            print(n)
            ## simple module
            print(old)
            setattr(model, n, new[c])
            c += 1


if __name__ == "__main__":

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square conv kernel
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.conv3 = nn.Conv2d(64, 512, 3)
            self.conv4 = nn.Conv2d(512, 64, 3)

            self.conv2 = nn.Conv2d(64, 16, 3)
            self.bn = nn.BatchNorm2d(16)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            print(x.shape)

            x = self.conv3(x)
            print(x.shape)

            x = self.conv4(x)
            x = self.conv2(x)

            print(x.shape)



            return self.bn(x)

    x = torch.randn(1,3,320,320).cuda()
    n3 = LeNet().cuda()

    print(n3(x).shape)
    torch.save(n3.state_dict(),'original.pth')

    print(n3)
    _,layer_data = summary(n3, input_size=(3, 320, 320))
    print(layer_data)
    # exit('layer_data')
    
    for lay in ['Conv2d','BatchNorm2d']:

        lay_list = [i for i in layer_data if i[0]==lay]
        print(lay_list)
        # exit('lay_list')

        if lay == 'Conv2d':
            replace_layers(n3, nn.Conv2d,[InvertedResidual(tup[1],tup[2],1,1) for tup in lay_list ])
        # elif lay == 'BatchNorm2d':
        #     replace_layers(n3, nn.BatchNorm2d,[nn.BatchNorm2d(tup[1]) for tup in lay_list ])

    # print(n3)
    # exit('n3')

    x = torch.randn(1,3,320,320).cuda()
    n3 = n3.cuda()
    print(n3(x).shape)

    # print(n3)
    torch.save(n3.state_dict(),'trasnformed.pth')
