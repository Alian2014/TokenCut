import sys
sys.path.append('./model')
import dino # model

import object_discovery as tokencut
import argparse
import utils
import bilateral_solver
import os

from shutil import copyfile
import PIL.Image as Image
import cv2
import numpy as np
from tqdm import tqdm

from torchvision import transforms
import metric
import matplotlib.pyplot as plt
import skimage
import torch

# Image transformation applied to all images
# 确保任何输入到 TokenCut 流程的图像，都和当初训练 ViT 模型时使用的 ImageNet 图像具有相似的数据分布
# 后续需要和 BoostAdapter 中所用的适用于 CLIP 的参数统一
ToTensor = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)),])


'''
get_tokencut_binary_map:                接收一个图片路径和相关模型参数，输出一个二值的分割图（掩码）和用于可视化的特征向量。

img_pth:                                字符串，需要进行分割的图片路径。

backbone:                               预训练好的骨干网络，即论文中提到的自监督视觉Transformer (ViT) 模型，用于提取特征。

patch_size:                             整数，将图片分割成的图像块 (patch) 的大小（例如 16，代表 16x16 像素）。

tau:                                    浮点数，即论文中的 相似度阈值 τ，在构建图时用于判断两个节点（图像块）之间是否存在强连接
'''


def get_tokencut_binary_map(img_pth, backbone,patch_size, tau) :
    # 函数使用 Pillow 库从 img_pth 路径打开图片，并用 .convert('RGB') 确保它是标准的3通道彩色图像
    I = Image.open(img_pth).convert('RGB')
    # 调用一个工具函数 utils.resize_pil 来调整图像尺寸,确保调整后的图像尺寸 (w, h) 能够被 patch_size 整除
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size)

    # 将图像转换为PyTorch张量，并进行标准化，增加一个维度，最后转移到GPU上
    tensor = ToTensor(I_resize).unsqueeze(0).cuda()
    # 将预处理好的张量送入 backbone 中，进行特征提取，生成一个特征向量
    feat = backbone(tensor)[0]

    # tokencut.ncut 函数利用上一步提取出的特征 feat 来构建一个图，然后应用归一化切割算法
    # 算法会计算出图的第二小特征向量 (eigvec)，得到一个二值的分割图 (bipartition)
    # 同时返回原始特征向量 eigvec
    seed, bipartition, eigvec = tokencut.ncut(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau)
    # bipartition: 二值分割图。这是一个二维数组或张量，其中的值（例如0和1）代表了每个图像块被归为背景还是前景。这就是论文中提到的粗糙掩码 (coarse mask)。
    # eigvec: 原始的特征向量。这个向量可以被重塑并可视化为一张热力图，直观地展示出模型认为图像的哪些区域是“显著”的，即论文中的“特征注意力 (eigen-attention)”。
    return bipartition, eigvec


'''
mask_color_compose:             将分割算法产生的掩码（mask）以指定的半透明颜色叠加到原始图像上，从而生成一个直观、漂亮的分割结果图。

org:                            NumPy数组格式的原始图像。

mask:                           NumPy数组格式的分割掩码，通常是get_tokencut_binary_map函数返回的bipartition。

mask_color:                     一个包含RGB三个值的列表，代表了叠加掩码时使用的颜色。默认值 [173, 216, 230] 是淡蓝色
'''


def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :
    # 这一步将输入的 mask 进行二值化处理,> 0.5 的为 True
    mask_fg = mask > 0.5
    # 为了不修改原始输入的图像数据，这里创建了一个原始图像 org 的副本 rgb。后续所有的修改都将在这个副本上进行。
    rgb = np.copy(org)
    # 颜色混合与叠加
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    # 最后，函数将修改后的NumPy数组 rgb 转换回 Pillow 库的图像（Image）对象，并将其返回。
    # 这个返回的图像就可以直接显示或保存
    return Image.fromarray(rgb)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--out-dir', type=str, help='output directory')

parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')

parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')


parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', 'DUT', None], help='which dataset?')

parser.add_argument('--nb-vis', type=int, default=100, choices=[1, 200], help='nb of visualization')

parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

args = parser.parse_args()
print (args)

## feature net

if args.vit_arch == 'base' and args.patch_size == 16:
    url = "/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'small' and args.patch_size == 16:
    url = "/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    feat_dim = 384
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#    resume_path = './model/dino_vitbase16_pretrain.pth' if args.patch_size == 16 else './model/dino_vitbase8_pretrain.pth'

#    feat_dim = 768
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#
#else :
#    resume_path = './model/dino_deitsmall16_pretrain.pth' if args.patch_size == 16 else './model/dino_deitsmall8_pretrain.pth'
#    feat_dim = 384
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)


msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
print (msg)
backbone.eval()
backbone.cuda()

if args.dataset == 'ECSSD' :
    args.img_dir = '../datasets/ECSSD/img'
    args.gt_dir = '../datasets/ECSSD/gt'

elif args.dataset == 'DUTS' :
    args.img_dir = '../datasets/DUTS_Test/img'
    args.gt_dir = '../datasets/DUTS_Test/gt'

elif args.dataset == 'DUT' :
    args.img_dir = '../datasets/DUT_OMRON/img'
    args.gt_dir = '../datasets/DUT_OMRON/gt'

elif args.dataset is None :
    args.gt_dir = None


print(args.dataset)

if args.out_dir is not None and not os.path.exists(args.out_dir) :
    os.mkdir(args.out_dir)

if args.img_path is not None:
    args.nb_vis = 1
    img_list = [args.img_path]
else:
    img_list = sorted(os.listdir(args.img_dir))

count_vis = 0
mask_lost = []
mask_bfs = []
gt = []
for img_name in tqdm(img_list) :
    if args.img_path is not None:
        img_pth = img_name
        img_name = img_name.split("/")[-1]
        print(img_name)
    else:
        img_pth = os.path.join(args.img_dir, img_name)
    
    bipartition, eigvec = get_tokencut_binary_map(img_pth, backbone, args.patch_size, args.tau)
    mask_lost.append(bipartition)

    output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition, sigma_spatial = args.sigma_spatial, sigma_luma = args.sigma_luma, sigma_chroma = args.sigma_chroma)
    mask1 = torch.from_numpy(bipartition).cuda()
    mask2 = torch.from_numpy(binary_solver).cuda()
    if metric.IoU(mask1, mask2) < 0.5:
        binary_solver = binary_solver * -1
    mask_bfs.append(output_solver)

    if args.gt_dir is not None :
        mask_gt = np.array(Image.open(os.path.join(args.gt_dir, img_name.replace('.jpg', '.png'))).convert('L'))
        gt.append(mask_gt)

    if count_vis != args.nb_vis :
        print(f'args.out_dir: {args.out_dir}, img_name: {img_name}')
        out_name = os.path.join(args.out_dir, img_name)
        out_lost = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut.jpg'))
        out_bfs = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_bfs.jpg'))
        #out_eigvec = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_eigvec.jpg'))

        copyfile(img_pth, out_name)
        org = np.array(Image.open(img_pth).convert('RGB'))

        #plt.imsave(fname=out_eigvec, arr=eigvec, cmap='cividis')
        mask_color_compose(org, bipartition).save(out_lost)
        mask_color_compose(org, binary_solver).save(out_bfs)
        if args.gt_dir is not None :
            out_gt = os.path.join(args.out_dir, img_name.replace('.jpg', '_gt.jpg'))
            mask_color_compose(org, mask_gt).save(out_gt)


        count_vis += 1
    else :
        continue

if args.gt_dir is not None and args.img_path is None:
    print ('TokenCut evaluation:')
    print (metric.metrics(mask_lost, gt))
    print ('\n')

    print ('TokenCut + bilateral solver evaluation:')
    print (metric.metrics(mask_bfs, gt))
    print ('\n')
    print ('\n')
