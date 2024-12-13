import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import random
import numpy as np
from tqdm import tqdm
from dataloader import myDataset
from model import DiQP
from utils.frame_utils import calcPSNR,reorder_image,batch_ssim
import warnings
import os
import csv
warnings.filterwarnings("ignore") 

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



qpPath= "Path to QPs frames"
rawPath = "Path to Raw frames"
model_path = './pretrained/checkpoint_AV1.pt' # or ./pretrained/checkpoint_HEVC.pt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

numOfFramesPerSeq = 300
extractingMethod = 'even'
totalQualities = [255]
seqNumbers=[5,11, 21, 23, 26 ] #these numbere are randomly selected sequences from SEPE8K for testing based on indices.csv from train.py and used for the paper.
img_size = 512

batchSize = 8

testset = myDataset(seqNumbers=seqNumbers,numOfFramesPerSeq=numOfFramesPerSeq,rawPath=rawPath,qpPath=qpPath,\
                extractingMethod=extractingMethod,totalQualities=totalQualities,cropSize= img_size,frac=.1,random_state=1234,augmentation=False,train=False,cropnos=120)
testloader =  DataLoader(testset,batch_size=batchSize,num_workers=10,pin_memory=True,shuffle=False)


model =  DiQP(img_size=img_size, embed_dim=15,depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], win_size=8, mlp_ratio=3., token_projection='linear', token_mlp='steff', shift_flag=True).to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)


model.eval()

psnr_clip_p,psnr_frame_p,psnr_channel_p,psnr_avg_p = 0.0, 0.0, 0.0, 0.0
psnr_clip_b,psnr_frame_b,psnr_channel_b,psnr_avg_b = 0.0, 0.0, 0.0, 0.0

ssim_avg_p = 0.0
ssim_avg_b = 0.0

with torch.no_grad():
    for j, (xCropped,yCropped,around,aheadCropped,aheadScaled,loc,log,decayFactor) in enumerate(tqdm(testloader)):
        xCropped = xCropped.to(device)
        aheadCropped = aheadCropped.to(device)
        aheadScaled = aheadScaled.to(device)
        around = around.to(device)
        yCropped = yCropped.to(device)
        decayFactor = decayFactor.to(device).view(-1,1,1,1,1)
        loc = loc.to(device).permute(1, 0, 2)
        
        with torch.cuda.amp.autocast():
            output = model(xCropped,around,aheadCropped,aheadScaled,loc,decayFactor)
        
        output = torch.clamp(output,0,1) 
        output= output.detach().cpu()
        yCropped = yCropped.detach().cpu()

        _psnr_clip_p,_psnr_frame_p,_psnr_channel_p,_psnr_avg_p = calcPSNR(output.cpu().numpy(),yCropped.cpu().numpy())
        _psnr_clip_b,_psnr_frame_b,_psnr_channel_b,_psnr_avg_b = calcPSNR(xCropped.cpu().numpy(),yCropped.cpu().numpy())


        psnr_clip_p,psnr_frame_p,psnr_channel_p,psnr_avg_p = \
              psnr_clip_p + _psnr_clip_p, psnr_frame_p + _psnr_frame_p,\
                  psnr_channel_p + _psnr_channel_p , psnr_avg_p + _psnr_avg_p
        
        psnr_clip_b,psnr_frame_b,psnr_channel_b,psnr_avg_b = \
              psnr_clip_b + _psnr_clip_b, psnr_frame_b + _psnr_frame_b,\
                  psnr_channel_b + _psnr_channel_b , psnr_avg_b + _psnr_avg_b

        _ssim_avg_p = batch_ssim(output.cpu().numpy(),yCropped.cpu().numpy())
        _ssim_avg_b = batch_ssim(xCropped.cpu().numpy(),yCropped.cpu().numpy())

        ssim_avg_p+=_ssim_avg_p
        ssim_avg_b+=_ssim_avg_b



        output = reorder_image(output)
        yCropped = reorder_image(yCropped)
        xCropped = reorder_image(xCropped)

        os.makedirs('testResults',exist_ok=True)

        for i,img in enumerate(output[0]):
            save_image(img, './testResults/o_{}_{}.png'.format(j,i+1))
        
        for i,img in enumerate(yCropped[0]):
            save_image(img, './testResults/y_{}_{}.png'.format(j,i+1))
        
        for i,img in enumerate(xCropped.detach().cpu()[0]):
            save_image(img, './testResults/x_{}_{}.png'.format(j,i+1))


psnr_clip_p,psnr_frame_p,psnr_channel_p,psnr_avg_p = \
        psnr_clip_p / (j+1),psnr_frame_p / (j+1),psnr_channel_p / (j+1),psnr_avg_p / (j+1)

psnr_clip_b,psnr_frame_b,psnr_channel_b,psnr_avg_b = \
        psnr_clip_b / (j+1),psnr_frame_b / (j+1),psnr_channel_b / (j+1),psnr_avg_b / (j+1)

ssim_avg_p = ssim_avg_p / (j+1)
ssim_avg_b = ssim_avg_b / (j+1)


log = [qpPath,totalQualities[0],model_path,psnr_avg_p,psnr_avg_b,ssim_avg_p,ssim_avg_b]


if os.path.isfile("./report.csv"):
    with open("./report.csv", 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(log)
else:
    with open("./report.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        field = ["Dataset","QP","model","PSNR Predicted", "PSNR Base", "SSIM Predicted","SSIM Base"]
        writer.writerow(field)
        writer.writerow(log)

