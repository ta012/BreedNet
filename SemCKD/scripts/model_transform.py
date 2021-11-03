import torch

if __name__ == '__main__':
    state_dict = torch.load('/home/tarun/Number_Theory/SemcKdWrapper_tony_Oct13/SemCKD/save/models/ResNet18_vanilla/resnet18-f37072fd.pth',map_location="cpu")
    torch.save({
        # 'epoch': model['epoch'],
        'model': state_dict, 
        # 'best_acc': model['best_acc1']
    }, '/home/tarun/Number_Theory/SemcKdWrapper_tony_Oct13/SemCKD/save/models/ResNet18_vanilla/teacher_transformed.pth')