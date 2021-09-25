
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import time
from .net_parser import parse_net 

#######################
from .layer_utils import InvertedResidual

#######################
class NetTrimmer:
	def __init__(self) -> None:
		pass


def replace_layers(model, old, new):
	c = 0  
	for n, module in model.named_children():
		if len(list(module.children())) > 0:
			## compound module, go inside it
			replace_layers(module, old, new)
			
		if isinstance(module, old):
			## simple module
			setattr(model, n, new[c])
			c += 1

def reduce_model_size(model=None,input_size=(3, 320, 320)):
	"""model size reduced, param count reduced, total calculation reduced, total params reduced,
	total forward time increase because of increase of sequential computation"""
	_,layer_data = parse_net(model, input_size=input_size)

	for lay in ['Conv2d','BatchNorm2d']:

		lay_list = [i for i in layer_data if i[0]==lay]

		if lay == 'Conv2d':
			replace_layers(model, nn.Conv2d,[InvertedResidual(tup[1],tup[2],1,1) for tup in lay_list ])
		# elif lay == 'BatchNorm2d':
		#     replace_layers(n3, nn.BatchNorm2d,[nn.BatchNorm2d(tup[1]) for tup in lay_list ])

	return model

def reduce_forward_time(model=None,input_size=(3, 320, 320)):
	"""total forward time decreased"""
	
	redn_factor = 0.8
	_,layer_data = parse_net(model, input_size=input_size)
	print(layer_data)
	new_layers = []
	for i,lay in enumerate(layer_data):
		if i == 0:
			new_layers.append((lay[0],lay[1],int(lay[2]*redn_factor)))
		elif i == len(layer_data)-1:
			new_layers.append((lay[0],int(lay[1]*redn_factor),lay[2]))
		else:
			new_layers.append((lay[0],int(lay[1]*redn_factor),int(lay[2]*redn_factor)))
	print(new_layers)

	layer_data = new_layers



	# for lay in layer_data[1:-1]:

	for lay in ['Conv2d','BatchNorm2d']:

		lay_list = [i for i in layer_data if i[0]==lay]


		if lay == 'Conv2d':
		    replace_layers(model, nn.Conv2d,[nn.Conv2d(tup[1],tup[2]) for tup in lay_list ])
		elif lay == 'BatchNorm2d':
		    replace_layers(n3, nn.BatchNorm2d,[nn.BatchNorm2d(tup[1]) for tup in lay_list ])

	return model
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

			x = self.conv3(x)

			x = self.conv4(x)
			x = self.conv2(x)

			return self.bn(x)

	x = torch.randn(1,3,320,320).cuda()
	n3 = LeNet().cuda()

	print(n3(x).shape)
	torch.save(n3.state_dict(),'original.pth')

	n3  = reduce_model_size(n3,(3, 320, 320))



	# print(n3)
	# exit('n3')

	x = torch.randn(1,3,320,320).cuda()
	n3 = n3.cuda()
	print(n3(x).shape)

	# print(n3)
	torch.save(n3.state_dict(),'trasnformed.pth')

	exit('done')


	from torchsummary import summary
	print(summary(n3, (3, 320, 320)))



	rep = 10
	resolution = 320
	device = 'cuda:0'

	time_l = []
	for i in range(rep):
		inp = torch.ones(1,3,resolution,resolution).to(device)

		start_time = time.time()
		out = n3(inp)
		# o = out.detach().cpu()
		# torch.cuda.synchronize()

		time_l.append(time.time()-start_time)

	   
	
	print("time taken after trimming",sum(time_l)/len(time_l))

	del n3


	n3 = LeNet().cuda()
	# print(summary(n3, (3, 320, 320)))
	time_l = []
	for i in range(rep):
		inp = torch.ones(1,3,resolution,resolution).to(device)

		start_time = time.time()
		out = n3(inp)
		# o = out.detach().cpu()
		# torch.cuda.synchronize()

		time_l.append(time.time()-start_time)

	   
	
	print("time taken without trimming",sum(time_l)/len(time_l))