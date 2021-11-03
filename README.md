# Breed-Net
<!-- PytorchHackathon 2021  -->
<!-- ![BreedNet](logo_br.jpg) -->

<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://demo.ailab.nolibox.com/) -->

<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="logo_new.jpg" alt="Logo" width="280" height="180">
  </a>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#what">What?</a>
    </li>
    <li><a href="#why">Why?</a></li>
    <li><a href="#how">How?</a></li>
    <li>
      <a href="#prerequisites">Prerequisites</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


## ```What?```
BreedNet is for creating a faster, lighter version of your trained computer vision neural network models.



## ```Why?```
We know that a larger network can easily fit the dataset than the smaller network because more the number of network connections, better the chance of finding an optimal solution. Using BreedNet you can create a faster, lighter version of your trained network for deploying in devices with low compute resources such as smart phones, smart watches and other embedded devices.


## ```How?``` 
BreedNet trims the input network as per the size ``` reduction factor ``` value. Then trains the trimmed model using the knowledge distillation with input model as teacher and trimmed model as stuent.

## ```Prerequisites```

* python
* anaconda
* pytorch

## ```Installation```
1. Clone the repo
   ```sh
   git clone https://github.com/ta94/BreedNet.git
   ```
2. Set up the environment
   ```sh
   conda env create -f breenet_env.yml
   ```

## ```Examples```

1. Torchvision Resnet18


```python
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
resnet_breednet = BreedNet(inp_net=model,redn_frac=0.5,gpu=False,train_epochs=1000,num_classes=100,input_size=(3,320,320))
print(resnet_breednet)

## trim input network
resnet_breednet.trim_net()
## the trimmed network is accessible using resnet_breednet.out_net
print("Size of trimmed net",model_size_estimater(resnet_breednet.out_net))

## train the trimmed network and 
## get best trimmed model and path of folder with best trimmed model torchscript and json with metrics information
net,torchscript_and_info_json_path = resnet_breednet.train(train_loader=train_loader,val_loader=val_loader)

```

2. Torchvision MobileNetV2 

```python
import torch
import torchvision

from breednet import BreedNet,model_size_estimater

#### dataset 
from SemCKD.dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

train_loader, val_loader = get_cifar100_dataloaders(batch_size=256,
                                                                num_workers=6)

### get input trained network
model = torchvision.models.mobilenet_v2(pretrained=False)
model.classifier[1]=torch.nn.Linear(in_features=1280, out_features=100, bias=True)
## cpu
#model.load_state_dict(torch.load('pretrained_models/mobilenetv2_cifar100-124-best.pth',map_location=torch.device("cpu")))
## gpu
model.load_state_dict(torch.load('pretrained_models/mobilenetv2_cifar100-124-best.pth'))

print("Size of Input net",model_size_estimater(model))

## breednet object creation
mobilenet_breednet = BreedNet(inp_net=model,redn_frac=0.75,gpu=True,train_epochs=1000,num_classes=100,input_size=(3,320,320))
print(mobilenet_breednet)

## trim input network
mobilenet_breednet.trim_net()
## the trimmed network is accessible using mobilenet_breednet.out_net

print("Size of trimmed net",model_size_estimater(mobilenet_breednet.out_net))

## train the trimmed network and 
## get best trimmed model and path of folder with best trimmed model torchscript and json with metrics information
net,torchscript_and_info_json_path = mobilenet_breednet.train(train_loader=train_loader,val_loader=val_loader)

```

<!-- ACKNOWLEDGMENTS -->
## ```Acknowledgments```

* [SemCKD](https://github.com/DefangChen/SemCKD)
* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)


<p align="right">(<a href="#top">back to top</a>)</p>
