import argparse
import glob
import os
import random
from math import inf, log
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
import scipy.io as sio
import pandas as pd
import base64

from estimate_gaussian_possion_noise import estimate_noise_variance
from models.network_dncnn import DnCNN as net
from models.unet import UNet_New
from models.unet_v1 import UNet
from utils.image_io import np_to_torch, pil_to_np, torch_to_np

parser = argparse.ArgumentParser()

parser.add_argument('--testset', type=str, default='cc', choices=['sidd', 'cc', 'polyu', 'DND', 
                                                'sidd_bench'])
parser.add_argument('--gpu_devices', default='3', type=str, help='gpu number')
parser.add_argument('--GMM', type=int, default=3, help='GMM number')
parser.add_argument('--lam', type=float, default=0.5, help='prior weight')
parser.add_argument('--alpha0', type=float, default=1.0, help='alpha_0')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_epoch', type=int, default=500, help='number of epoch')
parser.add_argument('--exp_weight', type=float, default=0.9, help='EMA')
opt, _ = parser.parse_known_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_devices

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

blind_iid_gaussian_denoisers = []
denoisers_pts = ['./model_zoo_unetdenoisers/UNet1.pth',
                 './model_zoo_unetdenoisers/UNet2.pth',
                 './model_zoo_unetdenoisers/UNet3.pth']

for no in range(opt.GMM):
    model = UNet(bias=False).cuda()
    model_no_path = denoisers_pts[no]
    
    model.load_state_dict(torch.load(model_no_path), strict=True)
    model.eval()
    
    for k, v in model.named_parameters():
        v.requires_grad = False
    blind_iid_gaussian_denoisers.append(model)
        
        
def denoising(noise_im, clean_im,  a0=1.0, b0=0.02, LR=1e-2, GMM_num=2, 
              total_step=400, exp_weight=0.99, lam=1):
    
    input_depth = noise_im.shape[0]
    
    noise_im_torch = torch.from_numpy(noise_im)[None, :].to(torch.float64)
    B, C, H, W = noise_im_torch.shape
    block_size = 4
    noise_im_patch = noise_im_torch.unfold(2, block_size, block_size).unfold(3, block_size, 
                     block_size).reshape(B, C, H//block_size, W//block_size, block_size**2).permute(0, 1, 4, 2, 3)

    var_est_list = np.zeros((block_size**2, C))
    for b in range(block_size**2):
        var_est_list[b] = estimate_noise_variance(noise_im_patch[:,:,b,...])
    
    var_mean = int(np.sqrt(np.mean(var_est_list))*255.0)

    if var_mean <= 10:
        lam = lam
    elif var_mean > 10 and var_mean <= 25:
        lam = 1.0
    else:
        lam = 1/lam
    
    x_net = UNet_New(input_depth, input_depth*2*GMM_num, down_sample_norm='batchnorm', 
                            up_sample_norm='batchnorm', chn_base=64, depth=4).cuda()

    phi_net = net(input_depth, input_depth*2*GMM_num, 32, 5).cuda()

    z_net = net(input_depth, input_depth*GMM_num, 32, 5).cuda()

    omg_net = net(input_depth, input_depth*GMM_num, 32, 5).cuda()
    
    optimizer = torch.optim.Adam([{"params": x_net.parameters(), "lr": LR},
                                  {"params": phi_net.parameters(), "lr": LR},
                                  {"params": z_net.parameters(), "lr": LR},
                                  {"params": omg_net.parameters(), "lr": LR},])  
    
    lr_schedualer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step, eta_min=5e-4)
    score = ScoreFunction.apply
    
    noise_im_torch = np_to_torch(noise_im)
    noise_im_torch = noise_im_torch.cuda()
    mean_avg = noise_im_torch.clone()
    
    log_max = log(1e4)
    log_min = log(1e-8)

    alpha_0 = torch.tensor([a0])[None, :, None, None, None].repeat(1,GMM_num,1,1,1).cuda() # prior parameter
    beta_0 = torch.tensor([b0])[None, :, None, None, None].repeat(1,GMM_num,1,1,1).cuda() # prior parameter
    d0 = torch.tensor([1.0])[None, :, None, None, None].repeat(1,GMM_num,1,1,1).cuda() # prior parameter

    for i in tqdm(range(total_step)):
            
        optimizer.zero_grad()

        # z output
        output_pi = z_net(noise_im_torch)
        B, C, H, W = output_pi.shape
        output_pi = output_pi.reshape(B, GMM_num, -1, H, W)
        output_pi = F.softmax(output_pi, dim=1)

        # omg output
        output_d = omg_net(noise_im_torch)
        output_d = torch.clamp(output_d, min=log_min, max=log_max)
        output_d = output_d.exp() # a_i, ..., a_k > 0
        B, C, H, W = output_d.shape
        output_d = output_d.reshape(B, GMM_num, -1, H, W)

        # Eq. 15
        digamma_a_m_a0 = torch.digamma(output_d) - torch.digamma(output_d.sum(dim=1, keepdim=True))
        kl_z = torch.sum(output_pi * ((output_pi+1e-8).log() - digamma_a_m_a0), dim=1).mean()

        # Eq. 16
        norm_posterior = torch.lgamma(output_d.sum(dim=1, keepdim=True)) - \
                            torch.lgamma(output_d).sum(dim=1, keepdim=True)
        norm_prior = torch.lgamma(d0.sum(dim=1, keepdim=True)) - \
                            torch.lgamma(d0).sum(dim=1, keepdim=True)
        kl_omg = torch.mean(norm_posterior - norm_prior) + torch.mean(((output_d - d0)*digamma_a_m_a0).sum(dim=1))

        # phi output
        output_alpha_beta = phi_net(noise_im_torch)
        output_alpha_beta = torch.clamp(output_alpha_beta, min=log_min, max=log_max)
        B, C, H, W = output_alpha_beta.shape
        output_alpha_beta = output_alpha_beta.reshape(B, GMM_num, -1, H, W)

        log_alpha, log_beta = torch.split(output_alpha_beta, [input_depth]*2, dim=2)

        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        alpha_div_beta = torch.exp(log_alpha - log_beta)
        
        # Eq. 14
        kl_phi = torch.sum( ( (alpha-alpha_0)*torch.digamma(alpha) + (torch.special.gammaln(alpha_0) - torch.special.gammaln(alpha))
                        + alpha_0*(log_beta - torch.log(beta_0)) + beta_0 * alpha_div_beta - alpha) * output_pi, dim=1).mean()
        
        # x output
        output_x = x_net(noise_im_torch)
        B, C, H, W = output_x.shape
        output_x = output_x.reshape(B, GMM_num, -1, H, W)
        mean, log_var = torch.split(output_x, [input_depth]*2, dim=2)
        log_var = torch.clamp(log_var, min=log_min, max=log_max)
        var = log_var.exp()
        sigma = var.sqrt()
        
        mc_samples = 1; tmp = torch.zeros_like(mean).repeat(mc_samples,1,1,1,1)
        z = torch.randn_like(tmp)*sigma + mean
        
        for j in range(GMM_num):
            blind_iid_gaussian_denoiser = blind_iid_gaussian_denoisers[j]
            with torch.no_grad():
                E_z = blind_iid_gaussian_denoiser(z[:, j])
            tmp[:, j] = tmp[:, j] + score(z[:, j], E_z.detach(), var[:, j].detach())

        kl_gauss = torch.sum((-0.5*log_var - tmp.mean(dim=0, keepdim=True))*output_pi, dim=1).mean()
        
        loss_rec = -torch.sum((0.5 * ( torch.digamma(alpha) - log_beta)
                              - 0.5 * ((mean - noise_im_torch[:, None, ...]).pow(2) + var) * alpha/beta) * output_pi, dim=1).mean()

        total_loss = loss_rec + lam*kl_gauss + kl_phi + kl_z + kl_omg
        
        
        total_loss.backward()
        optimizer.step()
        lr_schedualer.step()
        
        with torch.no_grad(): 
            mean = (mean*output_pi).sum(dim=1).mean(dim=0, keepdim=True)
            # using EMA
            mean_avg = mean_avg * exp_weight + mean.detach() * (1 - exp_weight)
            mean_np = torch_to_np(mean_avg)
            
            if i == (total_step-1) and clean_im is not None:
                psnr = compare_psnr(clean_im.transpose(1, 2, 0), mean_np.transpose(1, 2, 0), data_range=1)
                ssim = compare_ssim(clean_im.transpose(1, 2, 0), mean_np.transpose(1, 2, 0), channel_axis=-1, data_range=1)
            else:
                psnr = ssim = 0
        sigma_mean = sigma.mean().item()

    return psnr, ssim, mean_np.transpose(1, 2, 0), sigma_mean


if __name__ == "__main__":

    psnrs = []
    ssims = []

    result_base_folder = './output_scorediv+'
    
    result_folder = '{}'.format(opt.testset)

    result_folder = os.path.join(result_base_folder, result_folder)
    os.makedirs(result_folder, exist_ok=True)

    if opt.testset in ['cc', 'polyu', 'sidd']:

        if opt.testset == 'cc':
            path = './data/CC/'
            noises = sorted(glob.glob(path + '*real.png'))
            cleans = sorted(glob.glob(path + '*mean.png'))
            beta0 = 0.01

        elif opt.testset == 'polyu':
            path = './data/PolyU/'
            noises = sorted(glob.glob(path + '*real.JPG'))
            cleans = sorted(glob.glob(path + '*mean.JPG'))
            beta0 = 0.005
        
        elif opt.testset == 'sidd':
            lq_path = './data/SIDD_Val/noisy'
            hq_path = './data/SIDD_Val/GT'
            noises = sorted(glob.glob(os.path.join(lq_path,'*.png')))
            cleans = sorted(glob.glob(os.path.join(hq_path,'*.png')))
            beta0 = 0.02
        
        for step, (noise, clean) in enumerate(zip(noises, cleans)):
            
            noise_im = Image.open(noise)
            clean_im = Image.open(clean)
            
            noise_im_np = pil_to_np(noise_im)
            clean_im_np = pil_to_np(clean_im)
          
            psnr, ssim, denoised_im, sigma_mean = denoising(noise_im_np, clean_im_np, LR=opt.lr, a0=opt.alpha0, b0=beta0, exp_weight=opt.exp_weight,
            lam=opt.lam, GMM_num=opt.GMM, total_step=opt.n_epoch)
            
            if sigma_mean > 0.05: # 
                psnr, ssim, denoised_im, _ = denoising(noise_im_np, clean_im_np, LR=opt.lr, a0=opt.alpha0, b0=beta0, exp_weight=opt.exp_weight,
                    lam=opt.lam, GMM_num=opt.GMM, total_step=opt.n_epoch)
                    
            psnrs.append(psnr)
            ssims.append(ssim)
            
            img_name = noise.split('/')[-1].split('.')[0] + '_denoised'

            with open(os.path.join(result_folder, 'psnr.txt'), 'a') as f:
                print('Img Name: {}, PSNR: {}, SSIM: {}'.format(img_name, psnr, ssim), file=f, flush=True)
            
            # Image.fromarray((np.clip(denoised_im.squeeze(),0, 1)*255.0).astype(np.uint8), mode='RGB' if opt.testset != 'fmdd' else 'L' \
            #                ).save(os.path.join(result_folder, img_name + '.' + noise.split('.')[-1]))

        mean_psnr = sum(psnrs)/len(psnrs)
        mean_ssim = sum(ssims)/len(ssims)
        with open(result_folder + '/psnr.txt', 'a') as f:
            print('Mean PSNR: {}'.format(mean_psnr), file=f, flush=True)
            print('Mean SSIM: {}'.format(mean_ssim), file=f, flush=True)

    elif opt.testset == 'sidd_bench': 
        
        beta0 = 0.02; clean_im_np = None
        
        def my_srgb_denoiser(x):
            """TODO: Implement your own sRGB denoiser here."""
            x = np.float32(np.array(x)) / 255.0
            noise_im_np = x.transpose(2, 0, 1)
            
            _, _, denoised_im, sigma_mean = denoising(noise_im_np, clean_im_np, LR=opt.lr, a0=opt.alpha0, b0=beta0, exp_weight=opt.exp_weight,
                lam=opt.lam, GMM_num=opt.GMM, total_step=opt.n_epoch)
            if sigma_mean > 0.05: # 
                _, _, denoised_im, _ = denoising(noise_im_np, clean_im_np, LR=opt.lr, a0=opt.alpha0, b0=beta0, exp_weight=opt.exp_weight,
                    lam=opt.lam, GMM_num=opt.GMM, total_step=opt.n_epoch)
                    
            restored_patch = denoised_im

            restored = (restored_patch * 255.0).round().astype(np.uint8)
            return restored

        def array_to_base64string(x):
            array_bytes = x.tobytes()
            base64_bytes = base64.b64encode(array_bytes)
            base64_string = base64_bytes.decode('utf-8')
            return base64_string

        def base64string_to_array(base64string, array_dtype, array_shape):
            decoded_bytes = base64.b64decode(base64string)
            decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
            decoded_array = decoded_array.reshape(array_shape)
            return decoded_array

        input_file = './data/BenchmarkNoisyBlocksSrgb.mat'

        # Read inputs.
        key = 'BenchmarkNoisyBlocksSrgb'
        inputs = sio.loadmat(input_file)
        inputs = inputs[key]
        print(f'inputs.shape = {inputs.shape}')
        print(f'inputs.dtype = {inputs.dtype}')
        
        os.makedirs('./output_scorediv+/SIDD_bench', exist_ok=True)

        # Denoising.
        output_blocks_base64string = []
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                in_block = inputs[i, j, :, :, :]
                out_block = my_srgb_denoiser(in_block)
                assert in_block.shape == out_block.shape
                assert in_block.dtype == out_block.dtype
                out_block_base64string = array_to_base64string(out_block)
                output_blocks_base64string.append(out_block_base64string)

        # Save outputs to .csv file.
        output_file = os.path.join('./output_scorediv+/SIDD_bench', 'SubmitSrgb.csv')
        
        print(f'Saving outputs to {output_file}')
        output_df = pd.DataFrame()
        n_blocks = len(output_blocks_base64string)
        print(f'Number of blocks = {n_blocks}')
        output_df['ID'] = np.arange(n_blocks)
        output_df['BLOCK'] = output_blocks_base64string

        output_df.to_csv(output_file, index=False)

        # TODO: Submit the output file SubmitSrgb.csv at 
        # kaggle.com/competitions/sidd-benchmark-srgb-psnr
        print('TODO: Submit the output file SubmitSrgb.csv at')
        print('kaggle.com/competitions/sidd-benchmark-srgb-psnr')

        print('Done.')
        