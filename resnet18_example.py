import torch
import torchvision

from breednet import BreedNet,model_size_estimater
from SemCKD.dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

train_loader, val_loader = get_cifar100_dataloaders(batch_size=512,
                                                                num_workers=6)

### get input trained network
model = torchvision.models.resnet18()


model.fc = torch.nn.Linear(in_features=512, out_features=100, bias=True)
## for cpu
# model.load_state_dict(torch.load('pretrained_models/resnet18_torchvision_cifar100-196-best.pth',map_location=torch.device("cpu")))
## for cuda
model.load_state_dict(torch.load('pretrained_models/resnet18_torchvision_cifar100-196-best.pth'))

print("Size of Input net",model_size_estimater(model))

## breednet object creation
resnet_breednet = BreedNet(inp_net=model,redn_frac=0.5,gpu=True,train_epochs=1000,num_classes=100,input_size=(3,320,320))
print(resnet_breednet)

## trim input network
resnet_breednet.trim_net()
## the trimmed network is accessible using resnet_breednet.out_net
print("Size of trimmed net",model_size_estimater(resnet_breednet.out_net))

## train the trimmed network and 
## get best trimmed model and path of folder with best trimmed model torchscript and json with metrics information
net,torchscript_and_info_json_path = resnet_breednet.train(train_loader=train_loader,val_loader=val_loader)


