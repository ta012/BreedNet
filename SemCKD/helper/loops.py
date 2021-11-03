from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy, reduce_tensor
from trim.updated_forward import forward_update


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval() ## commented nan in eval

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) // opt.batch_size

    end = time.time()
    for idx, data in enumerate(train_loader):

        data_time.update(time.time() - end)

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data
            if opt.distill == 'semckd' and input.shape[0] < opt.batch_size:
                continue

        if (torch.cuda.is_available()) & (opt.gpu is not None):
            input = input.cuda()
            target = target.cuda()
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s = forward_update(model_s,input)
        with torch.no_grad():
            feat_t = forward_update(model_t,input)
            feat_t = [f.detach() for f in feat_t]

        logit_t = feat_t[-1]
        logit_s = feat_s[-1]
        feat_s = feat_s[:-1]
        feat_t = feat_t[:-1]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        
        # other kd beyond KL divergence
        
        if opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)                                                 
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        losses.update(loss.item(), input.size(0))

        metrics = accuracy(logit_s, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses, top1=top1, top5=top5,
                batch_time=batch_time, data_time=data_time))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg, data_time.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    # # switch to evaluate mod
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    n_batch = len(val_loader) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            input, target = batch_data

            if (torch.cuda.is_available()) & (opt.gpu is not None):
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = forward_update(model,input)[-1]

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            metrics = accuracy(output, target, topk=(1, 5))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
    
    return top1.avg, top5.avg, losses.avg
