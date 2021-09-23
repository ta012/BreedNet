import torch
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
            # self.conv = nn.Sequential(
            #     # dw
            #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU(inplace=True),
            #     # pw-linear
            #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            #     # nn.BatchNorm2d(oup),
            # )

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