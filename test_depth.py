import torch
import torch.nn.functional as F
import imageio
import os, argparse
from data_depth import test_dataset
from model.VGG16_depth import vgg16_depth

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = 'G:\Pycharm_Project\RGBD_saliency_dataset\\'

model = vgg16_depth()
model.load_state_dict(torch.load('G:\Pycharm_Project\CPD-depth\models\\vgg16_T=20\\.99_w.pth'))

model.cuda()
model.eval()

test_datasets = ['DUT-RGBD\\test_data', 'NJUD\\test_data', 'NLPR\\test_data', 'STEREO\\test_data']

for dataset in test_datasets:

    save_path = './results/VGG16_our/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '\\test_depth\\'
    gt_root = dataset_path + dataset + '\\test_masks\\'

    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        n, c, h, w = image.size()
        image = image.view(n,h,w,1).repeat(1,1,1,3)
        image = image.transpose(3,1)
        image = image.transpose(3,2)
        image = image.cuda()
        res = model(image)
        res = F.softmax(res, dim=1)

        res = res[0][1]
        res = res.data.cpu().numpy().squeeze()

        imageio.imwrite(save_path+name, res)
