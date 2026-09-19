import argparse
import glob
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as FT
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
from models.unet_v1 import UNet
from utils.utils.degredations import HDR, JPEG, NonlinearBlurOperator, PhaseRetrievalOperator

from utils.image_io import np_to_torch, pil_to_np, torch_to_np

def float2uint(img):
    img = np.clip(img, 0, 1)
    return np.uint8((img*255.0).round())


parser = argparse.ArgumentParser()

parser.add_argument('--testset', type=str, default='Set14', choices=['Set14', 'set5'])
parser.add_argument('--task', type=str, default='JPEG', choices=['HDR', 'PR', 'deblur_nl', 'JPEG'])
parser.add_argument('--n_epoch', type=int, default=2000, help='number of epoch')
opt, _ = parser.parse_known_args()

if opt.task == 'HDR':
    lam = 1e-3

elif opt.task == 'JPEG':
    qf = 10 # 压缩比为8时，图像质量退化严重
    lam = 1e-5
    
seed = 1314
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class ScoreFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, E_z, m2):
        ctx.save_for_backward(z, E_z, m2)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, E_z, m2 = ctx.saved_tensors
        grad_z = grad_output * (E_z - z)/m2.mean()
        return grad_z, None, None

# load blind iid Gaussian denoisers for computing scores
blind_iid_gaussian_denoisers = []
denoisers_pts = ['./model_zoo_unetdenoisers/UNet1.pth',
        './model_zoo_unetdenoisers/UNet2.pth',
        './model_zoo_unetdenoisers/UNet3.pth']
for no in range(3):
    model = UNet(bias=False).cuda()
    model_no_path = denoisers_pts[no]
    
    model.load_state_dict(torch.load(model_no_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    blind_iid_gaussian_denoisers.append(model)     
        
def Restoration(clean_im, total_step=400, task='deblur'):

    C, H, W = clean_im.shape
    clean_im_torch = np_to_torch(clean_im).cuda()
    
    if task == 'HDR':
        
        Forward = HDR()
        GMM_num = 3
        noise_im_torch = Forward.H(clean_im_torch)
        
        # mean = noise_im_torch.clone().unsqueeze(0).repeat(1, GMM_num, 1, 1, 1).requires_grad_() 
        mean = torch.randn_like(noise_im_torch).repeat(1, GMM_num, 1, 1, 1).requires_grad_() 
        log_var_ = torch.zeros(1, GMM_num, C, H, W, device='cuda').fill_(-5).requires_grad_()
    
       
    elif task == 'JPEG':
        Forward = JPEG(qf)
        GMM_num = 1
        
        noise_im_torch = Forward.H(clean_im_torch * 2 - 1)
        noise_im_torch = (noise_im_torch + 1) / 2
        # print(noise_im_torch.shape, clean_im_torch.shape)
        
        mean = noise_im_torch.clone().unsqueeze(0).repeat(1, GMM_num, 1, 1, 1).requires_grad_() 
        log_var_ = torch.zeros(1, GMM_num, C, H, W, device='cuda').fill_(-3).requires_grad_()
            
    
    noise_im = noise_im_torch.cpu().squeeze().permute(1, 2, 0).numpy()
    optimizer = torch.optim.Adam([mean, log_var_], lr= 0.01)  
    lr_schedualer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step, eta_min=1e-3)
    
    score = ScoreFunction.apply
    
    output_pi = 1/GMM_num; mc_samples = 1

    for i in tqdm(range(total_step)):
            
        optimizer.zero_grad()
        
        
        var = log_var_.exp()
        sigma = var.sqrt()
        
        tmp = torch.zeros_like(mean).repeat(mc_samples,1,1,1,1)
        z = torch.randn_like(tmp)*sigma + mean
        
        for j in range(GMM_num):
            blind_iid_gaussian_denoiser = blind_iid_gaussian_denoisers[j]
            with torch.no_grad():
                E_z = blind_iid_gaussian_denoiser(z[:, j])
            tmp[:, j] = tmp[:, j] + score(z[:, j], E_z, var.detach()) # iid var

        kl_gauss = torch.sum((-0.5*log_var_ - tmp.mean(dim=0, keepdim=True))*output_pi, dim=1).mean()
        
        # data terms
        num_samples = 10 # for data terms
        if task == 'HDR':
            tmp = torch.zeros_like(mean).repeat(num_samples,1,1,1,1)
            z = torch.randn_like(tmp)*sigma + mean
            loss_rec = (Forward.H(z) - noise_im_torch[None, ...]).abs().mean()
        elif task == 'JPEG':
            z =  torch.randn(size=(num_samples, mean.shape[1], mean.shape[2], mean.shape[3], mean.shape[4]), device=mean.device)*sigma + mean
            loss_rec = 0
            for i in range(GMM_num):
                loss_rec += (Forward.H(z[:, i]*2-1)/2+0.5 - noise_im_torch).pow(2).mean()
            loss_rec /= GMM_num
                  
        total_loss = loss_rec + lam*kl_gauss # lam=0.05*noise_std

        # print(loss_rec, kl_gauss)
        total_loss.backward()
        optimizer.step()
        lr_schedualer.step()
        
    with torch.no_grad(): 
        # Eq. 18
        mean_test = (mean*output_pi).sum(dim=1).mean(dim=0, keepdim=True)
        mean_np = torch_to_np(mean_test)

    return mean_np.transpose(1, 2, 0), noise_im


if __name__ == "__main__":

    psnrs = []
    ssims = []

    result_base_folder = './output_scoredvi+_nonlip'
    
    result_folder = '{}_{}'.format(opt.testset, opt.task)
    if opt.task == 'HDR':
        pass

    result_folder = os.path.join(result_base_folder, result_folder)
    os.makedirs(result_folder, exist_ok=True)
    
    cleans = sorted(glob.glob(os.path.join('/data0/cj/dataset', opt.testset, '*.png'))) \
        + sorted(glob.glob(os.path.join('/data0/cj/dataset', opt.testset, '*.bmp')))
    
    for step, clean in enumerate(cleans):

        clean_im = Image.open(clean)
        clean_im_np = pil_to_np(clean_im)# ; clean_im_np = clean_im_np[:, :256, :256]
        
        _, h, w = clean_im_np.shape; factor = 32; resolution = 256
        # x = F.pad(clean.float()/255.0, (0, factor-w%factor, 0, factor-h%factor))
        
        if h > resolution:
            clean_im_np = clean_im_np[...,:resolution,:]
        else:
            clean_im_np = np.pad(clean_im_np, ((0, 0), (0, resolution-h), (0, 0)))
        if w > resolution:
            clean_im_np = clean_im_np[...,:resolution]
        else:
            clean_im_np = np.pad(clean_im_np, ((0, 0), (0, 0), (0, resolution-w)))
        
        # H_pad = factor - H % factor; W_pad = factor - W % factor
        
        # clean_im_np_pad = np.pad(clean_im_np, ((0, 0), (0, H_pad), (0, W_pad)))
        # H_new, W_new, _ = clean_im_np_pad.shape
        
        denoised_im, noise_im = Restoration(clean_im_np, total_step=opt.n_epoch, task=opt.task)
        # denoised_im = denoised_im_pad[:H, :W, :]
        denoised_im = float2uint(denoised_im)
        
        psnr = compare_psnr(float2uint(clean_im_np.transpose(1, 2, 0)), denoised_im, data_range=255)
        ssim = compare_ssim(float2uint(clean_im_np.transpose(1, 2, 0)), denoised_im, channel_axis=-1, data_range=255)
        
        psnrs.append(psnr)
        ssims.append(ssim)
        
        img_name = clean.split('/')[-1].split('.')[0] + '_restore'
        img_name_deg = clean.split('/')[-1].split('.')[0] + '_deg'
        # noise_im = noise_im[:H, :W, :]
        Image.fromarray(denoised_im, mode='RGB').save(os.path.join(result_folder, img_name + '.' + clean.split('.')[-1]))
        Image.fromarray(float2uint(noise_im), mode='RGB').save(os.path.join(result_folder, img_name_deg + '.' + clean.split('.')[-1]))
        
    mean_psnr = sum(psnrs)/len(psnrs)
    mean_ssim = sum(ssims)/len(ssims)
    with open(result_folder + '/psnr.txt', 'a') as f:
        print('Mean PSNR: {}'.format(mean_psnr), file=f, flush=True)
        print('Mean SSIM: {}'.format(mean_ssim), file=f, flush=True)
    
