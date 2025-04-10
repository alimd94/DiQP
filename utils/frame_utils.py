import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from einops import rearrange
import cv2


def reorder_image(x):
    #batch_size = x.shape[0]
    x = rearrange(x, 'b c f h w -> b f c h w')
    return x


def calcPSNR(x, y, average=True):
    x = reorder_image(x)
    y = reorder_image(y)


    psnr_clip = 0.0
    psnr_frame = 0.0
    psnr_channel = 0.0

    for i,img in enumerate(x):
        psnr_clip += psnr(img, y[i])

        for j,f in enumerate(img):
            psnr_frame += psnr(f, y[i][j])

            for k,c in enumerate(f):
                psnr_channel += psnr(c, y[i][j][k])

    b,c,f,_,_ = x.shape

    
    if average:
        psnr_clip = psnr_clip/b 
        psnr_frame = psnr_frame / (b*f)
        psnr_channel = psnr_channel / (b*f*c)

        psnr_avg = (psnr_clip+psnr_frame+psnr_channel)/3

        return psnr_clip,psnr_frame,psnr_channel,psnr_avg 
    
    else:
        psnr_sum = psnr_clip+psnr_frame+psnr_channel
        
        return psnr_clip, psnr_frame, psnr_channel,psnr_sum



def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim_(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim_(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def batch_ssim(x,y, average=True, shape ='CF'):
    if shape == 'CF':
        x = rearrange(x, 'b c f h w -> b f h w c')
        y = rearrange(y, 'b c f h w -> b f h w c')
    elif shape == 'FC':
        x = rearrange(x, 'b f c h w -> b f h w c')
        y = rearrange(y, 'b f c h w -> b f h w c')
    else:
        raise ValueError('Wrong input image order.')
    
    b,f,_,_,c = x.shape

    ssim_batch = 0.0

    for i,clip in enumerate(x):
        ssim_clip = 0.0
        for j,frame in enumerate(clip):
            ssim_clip += calculate_ssim(frame, y[i][j])
        
        ssim_batch += ssim_clip/f
    
    ssim_batch = ssim_batch/b

    return ssim_batch