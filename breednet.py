"""
the general training framework
"""

from __future__ import print_function
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import re
import argparse
import time
import copy
import numpy

from trim.net_reducer import reduce_model_size
from trim.updated_forward import prepare_for_forward_update, forward_update
# from config import DEVICE
from SemCKD.helper.loops import train_distill as train, validate
from SemCKD.distiller_zoo import DistillKL, SemCKDLoss
from SemCKD.helper.util import adjust_learning_rate, save_dict_to_json, reduce_tensor
# from SemCKD.dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from SemCKD.models.util import  SelfA

# from utils import model_size_estimater

from params import args

total_time = time.time()
best_acc = 0

class BreedNet():
    def __init__(self,inp_net=None,redn_frac=None,gpu=True,train_epochs=None,num_classes=100,input_size=None):
        super(BreedNet, self).__init__()
        self.inp_net = inp_net
        self.out_net = copy.deepcopy(inp_net)

        self.redn_frac = redn_frac
        self.input_size = input_size
        self.num_classes = num_classes
        self.args = args
        self.args.gpu = gpu
        self.args.epochs = train_epochs
        if self.args.gpu:
            print('loading to cuda')
            self.inp_net = self.inp_net.cuda()
            self.out_net = self.out_net.cuda()    

    def trim_net(self,):
        """Trim the input network using """
        device = 'cpu'
        if self.args.gpu:
            device='cuda'

        reduce_model_size(self.out_net,self.input_size,self.redn_frac,device)
        self.out_net.apply(_kaiming_init_)


    def train(self,train_loader=None,val_loader=None):
        print("\nThe best trimmed model(torchscript) can be found in ",self.args.save_folder,'\n')

        self.out_net = train_student_net(model_t=self.inp_net,model_s=self.out_net,train_loader=train_loader,val_loader=val_loader,n_cls=self.num_classes,opt=args)



def train_student_net(model_t, model_s,train_loader,val_loader,n_cls,opt):
        
    global best_acc, total_time

    if opt.gpu:
        print("Use GPU: {} for training".format(opt.gpu))


    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)


    iterator = iter(train_loader)
    data, _ = iterator.next()

    if opt.gpu:
        data = data.cuda()

    handle_t = prepare_for_forward_update(model_t)
    handle_s = prepare_for_forward_update(model_s)

    # get the feature maps from the model
    feat_t = forward_update(model_t, data)
    feat_s = forward_update(model_s, data)

    


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    if opt.distill == 'semckd':
        assert len(feat_s) == len(feat_t) == 5
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        print("Number of feature maps used for SemCKD ",len(s_n))
        print("Teacher feature map shapes used for semckd : ", [i.shape for i in feat_t[1:-1]])
        print("Student feature map shapes used for semckd : ", [i.shape for i in feat_s[1:-1]])

        criterion_kd = SemCKDLoss()
        self_attention = SelfA(
            len(feat_s)-2, len(feat_t)-2, opt.batch_size, s_n, t_n)
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)     # SemKD knowledge distillation loss

    module_list.append(model_t)

    if torch.cuda.is_available():
        criterion_list.cuda()
        module_list.cuda()
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)


    if not opt.skip_validation:

        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
        print('teacher accuracy: ', teacher_acc)

    else:
        print('Skipping teacher validation.')

    # routine
    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()

        adjust_learning_rate(epoch, opt, optimizer)


        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss, data_time = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()

        print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}, Data {:.2f}'.format(
            epoch, opt.gpu, train_acc, train_acc_top5, time2 - time1, data_time))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        print('GPU %s validating' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate(
            val_loader,model_s,criterion_cls, opt)

        print(
            ' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))


        # save the best model
        if test_acc > best_acc:
            save_stu = copy.deepcopy(model_s)
            # save_stu = save_stu.to('cpu')
            ## save as torchscript to share network info
            save_stu = save_stu.eval()
            save_stu_copy = copy.deepcopy(save_stu)
            
            save_stu = torch.jit.script(save_stu)
            
            

            best_acc = test_acc
            save_file_ts = os.path.join(
                opt.save_folder, 'best_torchscript.pt')
            test_merics = {'test_loss': test_loss,
                            'test_acc': test_acc,
                            'test_acc_top5': test_acc_top5,
                            'best_acc': best_acc,
                            'epoch': epoch,
                            'pytorch_version':torch.__version__}

            save_dict_to_json(test_merics, os.path.join(
                opt.save_folder, "info.json"))
            print('saving the best model!')
            torch.jit.save(save_stu,save_file_ts)
            del save_stu
            torch.cuda.empty_cache()

    return save_stu_copy,opt.save_folder

def model_size_estimater(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return round(size_all_mb,2)

def _kaiming_init_(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # print(m.bias)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1) # alpha in doc equation 
        nn.init.constant_(m.bias, 0) # beta in doc equation



if __name__ == '__main__':
    import torchvision
    # model_teacher = torchvision.models.resnet18()
    # # model_teacher = torchvision.models.vgg16_bn()

    # model_teacher.fc = torch.nn.Linear(
    #     in_features=512, out_features=100, bias=True)

    # model_teacher.load_state_dict(torch.load(
    #     'resnet18_torchvision-196-best.pth'))

    model_path = 'pretrained_models/mobilenetv2-124-best.pth'

    model_teacher = torchvision.models.mobilenet_v2(pretrained=False)
    model_teacher.classifier[1]=torch.nn.Linear(in_features=1280, out_features=100, bias=True)
    model_teacher.load_state_dict(torch.load(model_path))

    

    model_s = copy.deepcopy(model_teacher)

    model_teacher = model_teacher.to(DEVICE)
    model_teacher.eval()

    model_s = model_s.to(DEVICE)

    # torch.save(model_teacher.state_dict(), 'original.pth')

    # n3 = copy.deepcopy(model_s)
    model_s = reduce_model_size(model_s, (3, 32, 32),opt.redn_frac)
    model_s = model_s.apply(_kaiming_init_)

    model_s = model_s.to(DEVICE)

    print("Input model size in MB",model_size_estimater(model_teacher))
    print("New model size in MB ",model_size_estimater(model_s))




    model_file = train_student_net(model_teacher, model_s)
    print(model_file)
    exit()
