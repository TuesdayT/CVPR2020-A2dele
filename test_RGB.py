import torch
import torch.nn.functional as F
import imageio
import os, argparse
from data_RGB import test_dataset
from model.VGG16_RGB import vgg16_RGB


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = 'G:\Pycharm_Project\RGBD_saliency_dataset\\' # test_datasets

model_RGB = vgg16_RGB()
model_RGB.load_state_dict(torch.load('G:\Pycharm_Project\LIPU-RGBD\models\\vgg16_RGB_adloss18\\42_w.pth'))

model_RGB.cuda()
model_RGB.eval()

test_datasets = ['DUT-RGBD\\test_data', 'NJUD\\test_data', 'NLPR\\test_data', 'STEREO\\test_data', 'LFSD', 'RGBD135','SSD']


for dataset in test_datasets:

    save_path = './results/2628_results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '\\test_images\\'
    gt_root = dataset_path + dataset + '\\test_masks\\'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):

        image, gt, name = test_loader.load_data()
        image = image.cuda()

        res, att_3, att_4, att_5 = model_RGB(image)
        res = F.softmax(res, dim=1)
        res = res[0][1]
        res = res.data.cpu().numpy().squeeze()

        imageio.imwrite(save_path+name, res)
