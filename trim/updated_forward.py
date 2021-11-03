
from types import new_class
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import time
# from torch.nn import Conv2d,BatchNorm2d,ReLU
from torch.nn import *
import time
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
import math

import pandas as pd
# from .net_parser import parse_net 

#######################
# from BreedNet.config import DEVICE


def get_feats_for_semckd(feat):

    if(len(feat) < 5):
        return "atleast 5 features are required"

    if(len(feat) == 5):
        return feat
    
    const_left = 0
    const_right = len(feat) - 1 
    
    temp_left = 1
    temp_right = len(feat)-2
    
    left = temp_left 
    right = temp_right 
    
    while(left < right):
        left = left + 1
        right = right - 1 
        
        if(left == right):
            center = left
    
    center = left
    
    return[feat[const_left],feat[temp_left],feat[center],feat[temp_right],feat[const_right]]

global out_l


def prepare_for_forward_update(model):

	model_name = str(model.__class__).split(".")[-1].split("'")[0]
	global out_l
	handle_l = []			

	def register_hook(module):
		global out_l
		def hook(module, input, output):
			global out_l
			
			class_name = str(module.__class__).split(".")[-1].split("'")[0]
			
			# print(class_name)
			# ['module','input_shape','output_shape','kernel_size']
			activation_list = ["ReLU","ReLU6"] ##@ts https://pytorch.org/docs/stable/nn.html fill remainng and put it in config, opt also shouldbe from config 
			if class_name in activation_list:
				out_l.append(output)

			elif class_name == model_name:
				out_l.append(output)		

		if (
			not isinstance(module, nn.Sequential)
			and not isinstance(module, nn.ModuleList)
		):
			handle_l.append(module.register_forward_hook(hook))

	# register hook
	model.apply(register_hook)

	return handle_l





def forward_update(model,image):
	global out_l
	out_l = []

	model(image)
	out_l = get_feats_for_semckd(out_l)

	# assert len(out_l) == 5

	return out_l


if __name__ == "__main__":
	import copy

	x = torch.randn(4,3,320,320).to(DEVICE)

	import torchvision
	
	n2 = torchvision.models.resnet18()
	# model_teacher = torchvision.models.vgg16_bn()


	n2.fc = torch.nn.Linear(in_features=512, out_features=100, bias=True)

	n2.load_state_dict(torch.load('resnet18_torchvision-100-regular.pth'))

	n3 = copy.deepcopy(n2)


	n2 = n2.to(DEVICE)
	n3 = n3.to(DEVICE)


	handle_l = prepare_for_forward_update(n2)

	print(len(handle_l))
	# exit()

	x = torch.randn(4,3,320,320).to(DEVICE)

	n2.eval()
	n3.eval()


	for _ in range(1000):
		
		
		


		with torch.no_grad():
			out = forward_update(n2,x)

			print(len(out))
			continue

			out = out[-1]

			# out = n2(x)
			out1 = n3(x)
			print(out)
			if not torch.equal(out,out1):
				exit(print("fail"))
			# print([i.shape for i in out])
			# print(out[-1])
			print('pass')

	exit()
	print(len(out))

	for h in handle_l[:-1]:
		h.remove()

	out = forward_update(n2,x)
	print(out[-1])
	exit()


	# for i in range(10):
	# 	out = forward_update(n2,x)
	# 	print(len(out))

	# # print(out)
	print(len(out)) # length of the output
	print([i.shape for i in out])
	