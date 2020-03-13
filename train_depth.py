import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import pdb, os, argparse
from datetime import datetime

from data_depth import get_loader
from utils import clip_gradient, adjust_lr
from model.VGG16_depth import vgg16_depth

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models

model = vgg16_depth()

vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
model.copy_params_from_vgg16_bn(vgg16_bn)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = 'G:\Pycharm_Project\\train_data-argument\\train_depth_negation\\'
gt_root = 'G:\Pycharm_Project\\train_data-argument\\train_masks\\'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

def cross_entropy2d(input, target, temperature=1, weight=None, size_average=True):
    target = target.long()
    n, c, h, w = input.size()
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    T = temperature
    loss = F.cross_entropy(input / T, target, weight=weight, size_average=size_average)
    # if size_average:
    #     loss /= mask.data.sum()
    return loss

MSE = torch.nn.MSELoss(size_average=True)

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack

        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        n, c, h, w = images.size()
        depth = images.view(n,h,w,1).repeat(1,1,1,3)
        depth = depth.transpose(3,1)
        depth = depth.transpose(3,2)

        dets = model(depth)

        loss = cross_entropy2d(dets, gts, temperature=20)

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))

    save_path = 'models/vgg16_depth/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path+ '.%d' % epoch+'_w.pth')

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
