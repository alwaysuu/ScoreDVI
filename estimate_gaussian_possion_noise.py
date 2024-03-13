'''
estimate the noise variance for general signal-dependent noise

This is the python implementation of the method presented in:

Xinhao Liu, Masayuki Tanaka, and Masatoshi Okutomi.
Practical signal-dependent noise parameter estimation from
a single noisy image. IEEE Transactions on Image Process-
ing, 23(10):4361–4371, 2014.

#

We note that this method is unreliable when local patches of the test image are pure white (very large intensity) 

'''


import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gamma
from scipy.optimize import minimize as minimize_scipy


def conv2mat(kernel, m, n):
    '''
    transform convolution operation to matrix multiplication
    '''

    H, W = kernel.shape
    HH = m-H+1; WW = n-W+1
    T = torch.zeros(HH*WW, m*n)

    k = 0
    for i in range(HH):
        for j in range(WW):
            for p in range(H):
                # print((i+p)*n+j+W)
                T[k, (i+p)*n+j:(i+p)*n+j+W] = kernel[p, :]
            
            k = k + 1

    return T

def im2col(img, block_size, step=1):
    '''
    image to column. we use tensor.unfold to implement this operation
    '''

    B, C, H, W = img.shape
    pH, pW = block_size

    # img = F.pad(img, [pH//2, pH//2, pW//2, pW//2], mode='replicate')

    img = img.unfold(2, pH, step=step).unfold(3, pW, step=step) # (H//pH, W//pW, pH, pW)
    img = img.reshape(B, C, -1, pH*pW).permute(0, 1, 3, 2) # (pH*pW, H//pH*W//pW)

    return img

def im2col_2dim(img, block_size, step=1):
    '''
    image to column. we use tensor.unfold to implement this operation
    '''

    H, W = img.shape
    pH, pW = block_size

    # img = F.pad(img, [pH//2, pH//2, pW//2, pW//2], mode='replicate')

    img = img.unfold(0, pH, step=step).unfold(1, pW, step=step) # (H//pH, W//pW, pH, pW)
    img = img.permute(1,0,3,2).reshape(-1, pH*pW).permute(1, 0) # (pH*pW, H//pH*W//pW)

    return img


def mle_loss(params, localMean, localVar):
    y = GSGNoiseModel(params[..., 0:1], params[..., 1:2], params[..., 2:3], localMean)
    loglik = localVar/(2*y) + torch.log(y)/2
    loglik = torch.sum(loglik)
    return loglik

def GSGNoiseModel(sigw, sigu, gamma, X):
    # print(sigu.shape, sigw.shape, X.shape)
    return X.pow(gamma*2) * sigu**2 + sigw**2

def mle_loss_np(params, localMean, localVar):
    y = GSGNoiseModel_np(params[0], params[1], params[2], localMean)
    loglik = localVar/(2*y) + np.log(y)/2
    loglik = np.sum(loglik)
    return loglik

def GSGNoiseModel_np(sigw, sigu, gamma, X):
    # print(sigu.shape, sigw.shape, X.shape)
    return np.power(X, gamma*2) * sigu**2 + sigw**2


def estimate_possion_gaussian_scipy(img, patchsize=7, conf=1-1e-6, itr=3):

    B, C, H, W = img.shape

    # img = (img - img.min(dim=1, keepdim=True)[0]) / (img.max(dim=1, keepdim=True)[0] - img.min(dim=1, keepdim=True)[0])

    kh = torch.tensor([[-0.5, 0, 0.5]], dtype=torch.float64)
    kv = kh.transpose(1,0)

    kh_4dim = kh[None, None, ...].repeat(C, 1, 1, 1)
    kv_4dim = kv[None, None, ...].repeat(C, 1, 1, 1)

    imgh = F.conv2d(img, kh_4dim, groups=C).pow(2)
    imgv = F.conv2d(img, kv_4dim, groups=C).pow(2)

    # conv to mat
    Dh = conv2mat(kh, patchsize, patchsize)
    Dv = conv2mat(kv, patchsize, patchsize)
    # print(Dh.shape, Dv.shape, Dh)
    DD = Dh.transpose(1, 0) @ Dh + Dv.transpose(1, 0) @ Dv

    r = torch.linalg.matrix_rank(DD).numpy()
    Dtr = torch.trace(DD).numpy()

    alpha0 = r / 2; scale0 = 2.0 * Dtr / r
    tau0 = gamma.ppf(conf, alpha0, scale=scale0)
    
    # print(r, Dtr, tau0)

    p_final = np.zeros((C, 3))
    Var_final = []

    for chn in range(C):

        X = im2col_2dim(img[0, chn], [patchsize, patchsize]) # B, C, pH*pW, H//pH*W//pW
        Xh = im2col_2dim(imgh[0, chn], [patchsize, patchsize-2])
        Xv = im2col_2dim(imgv[0, chn], [patchsize-2, patchsize])
        
        Xtr = torch.cat([Xh, Xv], dim=0).sum(dim=0, keepdim=True)
        Xave = X.mean(dim=0, keepdim=True) # B, C, H//pH*W//pW

        # noise axis estimation 
        cov = X @ X.permute(1, 0)/(X.shape[1]-1)
        _, V = torch.linalg.eigh(cov)
        # print(eigvalue, V[...,0:1])
        Xn = V[..., 0:1].permute(1, 0) @ X
        Xn = Xn ** 2


        # noise level estimation %%%%%
        fun = lambda x: mle_loss_np(x, Xave.squeeze().numpy(), Xn.squeeze().numpy())
        res = minimize_scipy(fun, (1e-6,0,0.0), bounds=((0, None), (0, None), (0, None))
            , method='Powell')
            
        sigw = res.x[0]
        sigu = res.x[1]
        gam = res.x[2]

        for i in range(itr):
        # weak texture selectioin
            tau = tau0 * GSGNoiseModel(sigw, sigu, gam, Xave)
            p = Xtr<tau
            chosen_col = p.nonzero()[:, 1]
            XX = torch.index_select(X, dim=1, index=chosen_col)
            XXave = torch.index_select(Xave, dim=1, index=chosen_col)
            # print(XX.shape, XXave.shape)
            
            # noise axis estimation 
            cov = XX @ XX.permute(1,0) /(XX.shape[1]-1)
            _, V = torch.linalg.eigh(cov)
            XXn = V[..., 0:1].permute(1, 0) @ XX
            XXn = XXn ** 2
            # Xn = Xn.squeeze(dim=2)
            
            # noise level estimation 
            
            fun = lambda x: mle_loss_np(x, XXave.squeeze().numpy(), XXn.squeeze().numpy())
            res = minimize_scipy(fun, (1e-6,0,0.0), bounds=((0, 100), (0, 100), (0, 100))
                            , method='Powell')
            # if not res.success:
            #    print('solution error!')

            sigw = res.x[0]
            sigu = res.x[1]
            gam = res.x[2]
        
        p_final[chn, :] = res.x
        Var_final.append(XXn)
        # print(res.x.exp())
    return p_final, Var_final

def estimate_noise_variance(img):
    _, Var_list = estimate_possion_gaussian_scipy(img)
    var = np.zeros(img.shape[1])
    
    for i in range(len(Var_list)):
        var[i] = torch.nan_to_num(Var_list[i].mean(), nan=0.0, posinf=1.0, neginf=1e-4).float().numpy()
        
    return var