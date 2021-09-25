import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/tony/work/work1/myprojects/BreedNet/')
from trim.net_reducer import reduce_model_size

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

        x = self.conv3(x)

        x = self.conv4(x)
        x = self.conv2(x)

        return self.bn(x)


if __name__ == "__main__":
    import torchvision


    # ########## LeNet ##############



    # x = torch.randn(1,3,320,320).cuda()
    # n3 = LeNet().cuda()
    # torch.save(n3.state_dict(),'original.pth')
    # print(summary(n3, (3, 320, 320)))
    # n3 = reduce_model_size(n3,(3,320,320))
    # n3 = n3.cuda()
    # torch.save(n3.state_dict(),'size_reduced.pth')

    # print(summary(n3, (3, 320, 320)))

    # ####################################

    n3 = torchvision.models.resnet18().cuda()
    torch.save(n3.state_dict(),'original.pth')
    n3 = reduce_model_size(n3,(3,320,320))
    n3 = n3.cuda()
    torch.save(n3.state_dict(),'size_reduced.pth')