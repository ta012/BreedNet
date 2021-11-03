
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import sleep

class SemCKDLoss(nn.Module):
    """     
            Class for the SemCKD           
            Cross-Layer Distillation with Semantic Calibration, 
            AAAI2021
    
    """
    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')
        
    def forward(self, s_value, f_target, weight):
        # print([i.shape for i in s_value])
        # print("[i.shape for i in s_value]")

        #weight gives you the shape  of these three
        bsz, num_stu, num_tea = weight.shape
        # print("\n\n\n")
        # print("weight shape")
        # print(bsz, num_stu, num_tea)
        # ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

        ind_loss = torch.zeros(bsz, num_stu, num_tea)
        # print(ind_loss.shape)

        # print("\n\n\n")
        # print("------------------s_value[0][0].shape--------------------")
        # print(s_value[0][0].shape)
        # print("\n\n\n")

        # sleep(2)

        # print("\n\n\n")
        # print("------------------t_value[0][0].shape--------------------")
        # print(f_target[0][0].shape)
        # print("\n\n\n")

        # sleep(2)

        
        for i in range(num_stu):
            for j in range(num_tea):
                # It can be explained as follows that is for each element in a batch there is MSE loss associated for a particular (i,j) combination
                # print()
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz,-1).mean(-1)
                

        # print("ind loss")
        # print(ind_loss.shape)
        # this loss is avg loss per batch per student
        # (weight * ind_loss).sum() is the scalar value 
        loss = (weight * ind_loss).sum()/( 1.0 * bsz * num_stu)
        # print((weight * ind_loss).sum())
        # print("\n\n\n")
        print("-----------------Semckd loss------------------------")
        print(loss.item())
        # print("\n\n\n")
        # sleep(2)

        return loss