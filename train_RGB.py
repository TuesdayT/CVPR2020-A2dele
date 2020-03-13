import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import math
import os, argparse
from datetime import datetime
import torch.nn as nn

from data_RGB import get_loader
from utils import clip_gradient, adjust_lr
from model.VGG16_RGB import vgg16_RGB
from model.VGG16_depth import vgg16_depth


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models

model_RGB = vgg16_RGB()
model_depth = vgg16_depth()

vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
model_RGB.copy_params_from_vgg16_bn(vgg16_bn)
model_depth.load_state_dict(torch.load('G:\Pycharm_Project\CPD-depth\models\\vgg16_T=20\\.99_w.pth'))

model_RGB.cuda()
model_depth.cuda()

params = model_RGB.parameters()

optimizer = torch.optim.Adam(params, opt.lr)

image_root = 'G:\Pycharm_Project\\train_data-argument\\train_images\\'
depth_root = 'G:\Pycharm_Project\\train_data-argument\\train_depth_negation\\'
gt_root = 'G:\Pycharm_Project\\train_data-argument\\train_masks\\'
train_loader = get_loader(image_root, depth_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()

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

def KD_KLDivLoss(Stu_output, Tea_output, temperature):
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(Stu_output/T, dim=1), F.softmax(Tea_output/T, dim=1))
    KD_loss = KD_loss * T * T
    return KD_loss

def Dilation(input):
    maxpool = nn.MaxPool2d(kernel_size=11, stride=1, padding=5)
    map_b = maxpool(input)
    return map_b

def train(train_loader, model_RGB, model_depth, optimizer, epoch):
    model_RGB.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images, depth, gts = pack

        images = Variable(images)
        depth = Variable(depth)
        gts = Variable(gts)
        images = images.cuda()
        depth = depth.cuda()
        gts = gts.cuda()

        n, c, h, w = images.size()
        depth = depth.view(n,h,w,1).repeat(1,1,1,3)
        depth = depth.transpose(3,1)
        depth = depth.transpose(3,2)

        dets_depth = model_depth(depth)
        dets, att_3, att_4, att_5 = model_RGB(images)

        loss_depth = cross_entropy2d(dets_depth.detach(), gts, temperature=20)
        loss_gt = cross_entropy2d(dets, gts, temperature=1)
        LIPU_loss = KD_KLDivLoss(dets, dets_depth.detach(), temperature=20)

        alpha = math.exp(-70*loss_depth)
        loss_adptative = (1-alpha)*loss_gt + alpha*LIPU_loss


        Dilation_depth = F.softmax(dets_depth, dim=1)
        Dilation_depth = Dilation(Dilation_depth[:,1,:,:].view(n, 1, h, w))

        loss_attention = CE(att_3, Dilation_depth.detach()) + \
                         CE(att_4, Dilation_depth.detach()) + \
                           CE(att_5, Dilation_depth.detach())

        loss = loss_adptative + loss_attention

        loss.backward()
        optimizer.step()

        clip_gradient(optimizer, opt.clip)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss_gt: {:.4f}, Loss_att: {:.4f}, Loss_LIPU: {:.4f}, alpha: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_gt.data, loss_attention.data, LIPU_loss.data, alpha))

    save_path = 'models/RGB_with_A2dele/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 1 == 0:
        torch.save(model_RGB.state_dict(), save_path+ '%d' % epoch+'_w.pth')


print("Let's go!")

for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model_RGB, model_depth, optimizer, epoch)
