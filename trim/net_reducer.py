
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
from .net_parser import parse_net 

#######################
from .layer_utils import InvertedResidual
# from BreedNet.config import DEVICE
#######################
DEVICE = 'cpu'

done_module_names = []
class NetTrimmer:
	def __init__(self) -> None:
		pass

def net_layer_names(model):
	name_dict = dict()
	for name, layer in model.named_modules():
		name_dict[str(name)]=layer

	return name_dict

def correct_layer_name(k):
	te = ''
	for n,a in enumerate(k.split('.')):
		try:
			int(a)
			te = te.rstrip('.')
			if n == len(k.split('.'))-1:
				te = te + '['+str(a)+']'
			else:
				te = te + '['+str(a)+']'+ '.'
		except:
			if n == len(k.split('.'))-1:
				te = te + a 
			else:
				te = te + a + '.'
	return te


def floor_3(n):
    if(n > 0):
        return math.ceil(n/3.0) * 3;
    else:
        exit('error')

def floor_3(n):
	return n

def replace_layers(model, input_size, batch_size=-1, device=None, dtypes=None,new_layers=[],nk=None):
	if dtypes == None:
		dtypes = [torch.FloatTensor]*len(input_size)
	
	layer_names = net_layer_names(model)

	summary_str = ''

	global count1,done_module_names
	count1 = 0

	def register_hook(module):
		global count1,done_module_names
		def hook(module, input, output):
			global count1,done_module_names
			
			class_name = str(module.__class__).split(".")[-1].split("'")[0]
			info_dict = dict()
			info_dict['module'] = str(module.__class__.__name__)


			info_dict["input_shape"] = list(input[0].size())
			info_dict["output_shape"] = list(output.size())



			if str(module.__class__.__name__) in ['Conv2d','BatchNorm2d','Linear']:
				
				for k,v in layer_names.items():
					module_name = None
					if v == module:

						if k in done_module_names:
							continue
						done_module_names.append(k)

						## if layer1.0.conv2 ---> layer1[0].conv2

						te = correct_layer_name(k)
						k = te

						module_name = k 
						
						break


				if module_name is not None:
					new_layers
					exec("model"+'.'+module_name +'='+ 'new_layers[count1]')
					model 
				count1 = count1 + 1

		if (
			not isinstance(module, nn.Sequential)
			and not isinstance(module, nn.ModuleList)
		):
			hooks.append(module.register_forward_hook(hook))

	# multiple inputs to the network
	if isinstance(input_size, tuple):
		input_size = [input_size]

	# batch_size of 2 for batchnormfloor_3
	x = [torch.rand(2, *in_size).type(dtype).to(device=device)
		 for in_size, dtype in zip(input_size, dtypes)]

	hooks = []
	

	# register hook
	model.apply(register_hook)

	# make a forward pass
	model(*x)


	# remove these hooks
	for h in hooks:
		h.remove()


def reduce_model_size(model=None,input_size=(3, 320, 320),redn_frac=0.5,device=None):
	"""model size reduced, param count reduced, total calculation reduced, total params reduced,
	total forward time increase because of increase of sequential computation"""
	layer_data = parse_net(model, input_size=input_size,device=device)


	lay_list = [i for i in layer_data]


	temp = []
	tempnet = []


	for k,c in enumerate(lay_list):
		if c['module']=='Conv2d':
			if (c['bias'] is not None) | (c['bias'] is not False):
				c['bias'] = True
		if(k!=0)&(k!=len(lay_list)-1):
			if c['module']=='Conv2d':
				temp.append((c['module'],int(c['input_shape'][1]),int(c['output_shape'][1]),c['kernel_size'][1],c['stride'][1]))

				if c['groups']!=1:
					c['groups'] = int(c['input_shape'][1]*redn_frac)
				tempnet.append(torch.nn.Conv2d(in_channels=int(c['input_shape'][1]*redn_frac), out_channels=int(c['output_shape'][1]*redn_frac), kernel_size=(c['kernel_size'][0],c['kernel_size'][1]), stride= (c['stride'][0],c['stride'][1]),padding = (c['padding'][0],c['padding'][1]),dilation=(c['dilation'][0],c['dilation'][1]),groups=c['groups'],bias=c['bias'],padding_mode=c['padding_mode']))#, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None))

			elif c['module'] == 'BatchNorm2d':
				assert c['input_shape'][1] == c['output_shape'][1], "error in trimming input network"
				temp.append((c['module'],int(c['input_shape'][1]),int(c['output_shape'][1])))
				tempnet.append(torch.nn.BatchNorm2d(int(c['input_shape'][1]*redn_frac)))

			# elif c['module'] == 'ReLU'
			elif c['module'] == 'Linear':
				temp.append((c['module'],int(c['input_shape'][1]),c['output_shape'][1]))
				tempnet.append(torch.nn.Linear(int(c['input_shape'][1]*redn_frac),int(c['output_shape'][1]*redn_frac)))

		if k==0:
			if c['module']=='Conv2d':
				temp.append((c['module'],c['input_shape'][1],int(c['output_shape'][1]),c['kernel_size'][1],c['stride'][1]))
				if c['groups']!=1:
					c['groups'] = int(c['input_shape'][1])
				tempnet.append(torch.nn.Conv2d(in_channels=int(c['input_shape'][1]), out_channels=int(c['output_shape'][1]*redn_frac), kernel_size=(c['kernel_size'][0],c['kernel_size'][1]), stride= (c['stride'][0],c['stride'][1]),padding = (c['padding'][0],c['padding'][1]),dilation=(c['dilation'][0],c['dilation'][1]),groups=c['groups'],bias=c['bias'],padding_mode=c['padding_mode']))#, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None))
			else:
				temp.append((c['module'],int(c['output_shape'][1])))
				exit("Warning Layers starts with ",c['module'],"not Conv2d")

		if k==len(lay_list)-1:
			if c['module']=='Conv2d':	
				temp.append((c['module'],int(c['input_shape'][1]),c['output_shape'][1],c['kernel_size'][1],c['stride'][1]))
				if c['groups']!=1:
					c['groups'] = int(c['input_shape'][1]*redn_frac)
				tempnet.append(torch.nn.Conv2d(in_channels=int(c['input_shape'][1]*redn_frac), out_channels=int(c['output_shape'][1]), kernel_size=(c['kernel_size'][0],c['kernel_size'][1]), stride= (c['stride'][0],c['stride'][1]),padding = (c['padding'][0],c['padding'][1]),dilation=(c['dilation'][0],c['dilation'][1]),groups=c['groups'],bias=c['bias'],padding_mode=c['padding_mode']))#, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None))

			elif c['module'] == 'Linear':
				temp.append((c['module'],int(c['input_shape'][1]),c['output_shape'][1]))
				tempnet.append(torch.nn.Linear(int(c['input_shape'][1]*redn_frac),c['output_shape'][1]))

			else:
				assert c['input_shape'][1] == c['output_shape'][1]
				temp.append((c['module'],int(c['input_shape'][1]*redn_frac)))


	replace_layers(model,input_size,batch_size=-1, device=device,dtypes=None,new_layers=tempnet,nk=0)
	model = model.to(device)