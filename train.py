import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist  
import torch.optim as optim
import random
import numpy as np
import os
from tqdm import tqdm
import csv
from sklearn.model_selection import ShuffleSplit

from warmup_scheduler import GradualWarmupScheduler
from timm.utils import NativeScaler
from losses import CharbonnierLoss

from dataloader import myDataset
from model import DiQP
from utils.frame_utils import calcPSNR

import warnings
warnings.filterwarnings("ignore") 

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self,model,train_data,val_data,optimizer,scheduler,gpu_id,epochNos):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossFn = CharbonnierLoss().to(gpu_id)
        self.loss_scaler = NativeScaler()
        self.model = DDP(model, device_ids=[gpu_id])
        self.eval_now = len(self.train_data)//4
        self.epochNos = epochNos
        self.bestIter = 0
        self.bestEpoch = 0
        self.bestPSNR = 0
        self.resume = True

    def _logger(self,log,path,train=False):
        if train:
            if os.path.isfile(path):
                with open(path, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(log)
            else:
                with open(path, 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    field = ["iter","loss_avg","psnr_avg", "seqNum", "middle","qp","lr_scheduler"]
                    writer.writerow(field)
                    writer.writerow(log) 
        else:
            if os.path.isfile(path):
                with open(path, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(log)
            else:
                with open(path, 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    field = ["epoch","iter", "clip","frame","channel","avg","best_epoch","best_iter","best_psnr"]
                    writer.writerow(field)
                    writer.writerow(log)

    def _run_batch(self,iter, xCropped,yCropped,around,aheadCropped,aheadScaled,loc,decayFactor,log,lossScaler): 

        loss_total = 0.0
        (seqNum,middle,qp) = log

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = self.model(xCropped,around,aheadCropped,aheadScaled,loc,decayFactor)
            loss = self.lossFn(output, yCropped)

        lossScaler(loss, self.optimizer,parameters=self.model.parameters())
        loss_total += loss

        if self.gpu_id == 0 and (iter+1) % 1000 == 0:
            output = torch.clamp(output,0,1)  
            psnr_clip,psnr_frame,psnr_channel,psnr_avg = calcPSNR(output.detach().cpu().numpy(),yCropped.detach().cpu().numpy())
            self._logger((iter,loss_total.detach().cpu(),psnr_avg,seqNum,middle,qp,self.scheduler.get_lr()),train=True,path='./train.csv')

    def _validate(self,valid_loader,):

        def run_validate(valid_loader):

            val_clip,val_frame,val_channel,val_avg = 0.0, 0.0, 0.0, 0.0

            count = torch.zeros(1, dtype=torch.float32,).to(self.gpu_id)

            with torch.no_grad():
                for i, (xCropped,yCropped,around,aheadCropped,aheadScaled,loc,log,decayFactor) in enumerate(valid_loader):

                    xCropped = xCropped.to(self.gpu_id)
                    aheadCropped = aheadCropped.to(self.gpu_id)
                    aheadScaled = aheadScaled.to(self.gpu_id)
                    around = around.to(self.gpu_id)
                    yCropped = yCropped.to(self.gpu_id)
                    loc = loc.to(self.gpu_id).permute(1, 0, 2)
                    decayFactor = decayFactor.to(self.gpu_id).view(-1,1,1,1,1)

                    with torch.cuda.amp.autocast():
                        output = self.model(xCropped,around,aheadCropped,aheadScaled,loc,decayFactor)

                    output = torch.clamp(output,0,1) 
                    psnr_clip,psnr_frame,psnr_channel,psnr_avg = calcPSNR(output.detach().cpu().numpy(),yCropped.detach().cpu().numpy())

                    val_clip += torch.Tensor([psnr_clip]).to(self.gpu_id)
                    val_frame += torch.Tensor([psnr_frame]).to(self.gpu_id)
                    val_channel += torch.Tensor([psnr_channel]).to(self.gpu_id)
                    val_avg += torch.Tensor([psnr_avg]).to(self.gpu_id)

                    count += 1

            return val_clip,val_frame,val_channel,val_avg, count

        self.model.eval()
        val_clip,val_frame,val_channel,val_avg, count = run_validate(valid_loader)

        dist.barrier()
        dist.all_reduce(val_clip, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(val_frame, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(val_channel, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(val_avg, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(count, dist.ReduceOp.SUM, async_op=False)

        avg_val_clip = float((val_clip / count).detach().cpu())
        avg_val_frame = float((val_frame / count).detach().cpu())
        avg_val_channel = float((val_channel / count).detach().cpu())
        avg_val = float((val_avg / count).detach().cpu())

        self.model.train()
        torch.cuda.empty_cache()

        return avg_val_clip, avg_val_frame, avg_val_channel, avg_val

    def _run_epoch(self,epoch,lossScaler):

        for i, (xCropped,yCropped,around,aheadCropped,aheadScaled,loc,log,decayFactor) in enumerate(self.train_data):

            xCropped = xCropped.to(self.gpu_id)
            aheadCropped = aheadCropped.to(self.gpu_id)
            aheadScaled = aheadScaled.to(self.gpu_id)
            around = around.to(self.gpu_id)
            yCropped = yCropped.to(self.gpu_id)
            loc = loc.to(self.gpu_id).permute(1, 0, 2)
            decayFactor = decayFactor.to(self.gpu_id).view(-1,1,1,1,1)

            self._run_batch(i,xCropped,yCropped,around,aheadCropped,aheadScaled,loc,decayFactor,log,lossScaler)

            if (i+1)%1000 == 0 and i>0:
                avg_val_clip, avg_val_frame, avg_val_channel, avg_val=self._validate(self.val_data)
                if self.gpu_id == 0:
                    if self.bestPSNR < avg_val:
                        self.bestPSNR = avg_val
                        self.bestIter = i
                        self.bestEpoch = epoch
                        self._save_checkpoint(epoch,i)

                    self._logger((epoch,i,avg_val_clip, avg_val_frame, avg_val_channel, avg_val,self.bestEpoch,self.bestIter,self.bestPSNR),path='./validation.csv')
            if i == 2:
                break
            
    def _save_checkpoint(self,epoch,i):
        ckp = self.model.module.state_dict()
        optz = self.optimizer.state_dict()
        torch.save(ckp,  "./checkpoint_{}_{}.pt".format(epoch,i))
        torch.save(optz, "./opt{}_{}.pt".format(epoch,i))

    def train(self,lossScaler):
        for epoch in tqdm(range(self.epochNos)):
            self._run_epoch(epoch,lossScaler)
            self.scheduler.step()

def prepare_dataloader(trainset,valset, batch_size):
    return (DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=DistributedSampler(trainset)),
            DataLoader(
        valset,
        batch_size=batch_size*4,
        sampler=DistributedSampler(valset)))

def main(rank, world_size, total_epochs, batch_size):
    ddp_setup(rank, world_size)

    random_state = 1234

    qpPath= "Path to QP frames"
    rawPath = "PAth to Raw frames"

    cv = ShuffleSplit(n_splits=1, test_size=1/8, random_state=0)
    trainIdx,testIdx = next(iter(cv.split(np.arange(1,41)))) # if we consider sequences are like [1,2,3,...,40]
    trainSeqNumbers = np.arange(1,41)[trainIdx][:30]
    valSeqNumbers = np.arange(1,41)[trainIdx][30:]


    with open('./indices.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(["val"])
        writer.writerow(valSeqNumbers)
        writer.writerow(["train"])
        writer.writerow(trainSeqNumbers)

    numOfFramesPerSeq = 300
    extractingMethod = 'even'
    totalQualities = [] #list of QPs

    trainFrac = 1 #on how much of data model should be trianed 
    valFrac = 1 #on how much of data model should be validated 

    crop_size = 512

    learning_rate = 0.0002


    trainset = myDataset(seqNumbers=trainSeqNumbers,numOfFramesPerSeq=numOfFramesPerSeq,rawPath=rawPath,qpPath=qpPath,\
                         extractingMethod=extractingMethod,totalQualities=totalQualities,cropSize= crop_size,frac=trainFrac,random_state=random_state,augmentation=False,train=True)

    valset = myDataset(seqNumbers=valSeqNumbers,numOfFramesPerSeq=numOfFramesPerSeq,rawPath=rawPath,qpPath=qpPath,\
                       extractingMethod=extractingMethod,totalQualities=255,cropSize= crop_size,frac=valFrac,random_state=random_state,augmentation=False,train=False)

    model =  DiQP(crop_size=crop_size, embed_dim=15,depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], win_size=8, mlp_ratio=3., token_projection='linear', token_mlp='steff', shift_flag=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-8, weight_decay=0.02)


    warmup_epochs = 1
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    lossScaler = NativeScaler()

    train_data,val_data = prepare_dataloader(trainset,valset, batch_size)
    trainer = Trainer(model, train_data,val_data, optimizer,scheduler, rank,total_epochs)
    torch.cuda.empty_cache()
    trainer.train(lossScaler)

    dist.destroy_process_group()

if __name__ == "__main__":

    total_epochs = 1
    batch_size = 3

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, batch_size), nprocs=world_size)
