import argparse
import glob
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as FT
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
from models.unet_v1 import UNet

from utils.image_io import np_to_torch, pil_to_np, torch_to_np

def float2uint(img):
    img = np.clip(img, 0, 1)
    return np.uint8((img*255.0).round())


parser = argparse.ArgumentParser()

parser.add_argument('--testset', type=str, default='set5', choices=['Set14', 'set5'])
parser.add_argument('--task', type=str, default='deblur', choices=['deblur', 'inpaint'])
parser.add_argument('--n_epoch', type=int, default=2000, help='number of epoch')
opt, _ = parser.parse_known_args()

if opt.task == 'deblur':
    kernel_size = 9; kernel_std = [20, 1]; noise_std = 0.02
elif opt.task == 'inpaint':
    mask_type = 'text'

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
    if task == 'deblur':
        GMM_num = 3
        noise_im_torch = FT.gaussian_blur(clean_im_torch, kernel_size, kernel_std) + torch.randn_like(clean_im_torch) * noise_std
        
        mean = noise_im_torch.clone().unsqueeze(0).repeat(1, GMM_num, 1, 1, 1).requires_grad_() 
        log_var_ = torch.zeros(1, GMM_num, C, H, W, device='cuda').fill_(-3).requires_grad_()
        
    elif task == 'inpaint':
        GMM_num = 1
        _, C, H, W = clean_im_torch.shape
        
        if mask_type == 'text':
            mask_np = np.array(Image.open("./mask/text.bmp")).astype(np.float32).mean(axis=-1)/255.0
        elif mask_type == 'lorem3': # follow DDRM
            mask_np = np.load("./mask/text.npy")

        mask_np = cv2.resize(mask_np, dsize=(W,H))

        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).cuda()
        noise_im_torch = clean_im_torch * mask
        mean = torch.rand(1, GMM_num, C, H, W, requires_grad=True, device='cuda')

    
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
        
        if task == 'deblur':
            # eq 29
            mean_convolve = FT.gaussian_blur(mean.squeeze(0), kernel_size, kernel_std).unsqueeze(0)
            loss_rec = torch.sum(0.5 * (mean_convolve - noise_im_torch[:, None, ...]).pow(2) * output_pi, dim=1).mean()
            loss_rec += var.mean() * 0.5 # in practice, we found that setting |k|^2 to 1 results in good performance
            total_loss = loss_rec + 0.05*noise_std*kl_gauss # lam=0.05*noise_std
        
        elif task == 'inpaint':
            # eq 30
            mean_masked = mask * mean
            loss_rec = torch.sum(0.5 * ((mean_masked - noise_im_torch[:, None, ...]).pow(2)) * output_pi, dim=1).mean()
            loss_rec += ((mask)*var).mean() * 0.5 
            total_loss = loss_rec + 0.02*kl_gauss  # lam=0.02
                 
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

    result_base_folder = './output_scoredvi+_lip'
    
    result_folder = '{}_{}'.format(opt.testset, opt.task)
    if opt.task == 'deblur':
        result_folder += '_k{}s{}n{}'.format(kernel_size, kernel_std, noise_std)
    elif opt.task == 'inpaint':
        result_folder += '_{}'.format(mask_type)

    result_folder = os.path.join(result_base_folder, result_folder)
    os.makedirs(result_folder, exist_ok=True)
    
    cleans = sorted(glob.glob(os.path.join('/data0/cj/dataset', opt.testset, '*.png'))) \
        + sorted(glob.glob(os.path.join('/data0/cj/dataset', opt.testset, '*.bmp')))
    
    for step, clean in enumerate(cleans):

        clean_im = Image.open(clean)
        clean_im_np = pil_to_np(clean_im)
        C, H, W = clean_im_np.shape
        
        H_pad = 16 - H % 16; W_pad = 16 - W % 16
        
        clean_im_np_pad = np.pad(clean_im_np, ((0, 0), (0, H_pad), (0, W_pad)))
        H_new, W_new, _ = clean_im_np_pad.shape
        
        denoised_im_pad, noise_im = Restoration(clean_im_np_pad, total_step=opt.n_epoch, task=opt.task)
        denoised_im = denoised_im_pad[:H, :W, :]
        denoised_im = float2uint(denoised_im)
        
        psnr = compare_psnr(float2uint(clean_im_np.transpose(1, 2, 0)), denoised_im, data_range=255)
        ssim = compare_ssim(float2uint(clean_im_np.transpose(1, 2, 0)), denoised_im, channel_axis=-1, data_range=255)
        
        psnrs.append(psnr)
        ssims.append(ssim)
        
        # img_name = clean.split('/')[-1].split('.')[0] + '_restore'
        # img_name_deg = clean.split('/')[-1].split('.')[0] + '_deg'
        # noise_im = noise_im[:H, :W, :]
        # Image.fromarray(denoised_im, mode='RGB').save(os.path.join(result_folder, img_name + '.' + clean.split('.')[-1]))
        # Image.fromarray(float2uint(noise_im), mode='RGB').save(os.path.join(result_folder, img_name_deg + '.' + clean.split('.')[-1]))
        
    mean_psnr = sum(psnrs)/len(psnrs)
    mean_ssim = sum(ssims)/len(ssims)
    with open(result_folder + '/psnr.txt', 'a') as f:
        print('Mean PSNR: {}'.format(mean_psnr), file=f, flush=True)
        print('Mean SSIM: {}'.format(mean_ssim), file=f, flush=True)
    