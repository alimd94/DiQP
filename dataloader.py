from torch.utils.data import Dataset
import os 
import random
import pandas as pd
import itertools
import numpy as np
import cv2
import threading

from albumentations import (
   Compose,OneOf,ReplayCompose,Flip,
   RandomRotate90)
import csv
from einops import rearrange


class myDataset(Dataset):
    def _loadImages(self,frameNos,seqNum,qp,step=50,count=1): #method for loading frames and put them into order since whe used threading for loading them.
        frames = []
        images = []
        types= []

        qpPath = os.path.join(self.qpPath,"{:03d}".format(seqNum),"QP-{}".format(str(qp)))
        rawPath = os.path.join(self.rawPath,"{:03d}".format(seqNum))

        def loadImage(*arg):
            file_name,frame,typ = arg
            image =cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            types.append(typ)
            if image.shape[2] == 4:
                images.append(image[:,:,:3])
            else:
                images.append(image)

        offsetW = random.randint(0,(self.widthOrg - self.VisibleWindow['width']))
        offsetH = random.randint(0,(self.heightOrg - self.VisibleWindow['height']))
        left = offsetW
        right = offsetW + self.VisibleWindow['width']
        top = offsetH
        bottom = offsetH + self.VisibleWindow['height']

        imagesArg = []

        for i,f in enumerate(frameNos):
            image_filename =os.path.join(qpPath,"{:03d}.png".format(f))
            imagesArg.append([image_filename,f,1])

            image_filename =os.path.join(rawPath,"{:03d}_8K.png".format(f))
            imagesArg.append([image_filename,f,0])
        
        if frameNos[1] < self.numOfFramesPerSeq - step:
            lookahead = [frameNos[-1] + step * (i+1) for i in range(count) if frameNos[-1] + step * (i+1) < self.numOfFramesPerSeq-1 ]
        else:
            lookahead = [self.numOfFramesPerSeq -1]

        for f in lookahead:
            image_filename =os.path.join(qpPath,"{:03d}.png".format(f)) 
            imagesArg.append([image_filename,f,-1])

        threads = []
        for arg in imagesArg:
            thread = threading.Thread(target=loadImage, args=arg,)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
                                    
        images = np.array(images)[:,top:bottom,left:right,:]
        
        if len(images) != len(types) != len(frames) :
            raise("Somthing missing!")
                
        frameNos = np.array(frameNos)
        lookahead = np.array(lookahead)

        xImages,yImages,lookaheadImages = [None,None,None],[None,None,None],[np.zeros((self.VisibleWindow['height'],self.VisibleWindow['width'],3),dtype=np.uint8) for i in range(count)]
    
        for i,typ in enumerate(types):
            if typ == 1:
                xImages[int(np.argwhere(frameNos==frames[i]))]=images[i]
            elif typ == 0:
                yImages[int(np.argwhere(frameNos==frames[i]))]=images[i]
            elif typ == -1:
                lookaheadImages[int(np.argwhere(lookahead==frames[i]))]=images[i]
            else:
                raise("type error")

        return np.array(xImages,dtype=np.uint8),np.array(yImages,np.uint8),np.array(lookaheadImages,np.uint8),np.array(frameNos),np.array(types),(left,top)

    def _createSampleSpace(self,seqNumbers,numOfFrames,extractingMethod,totalQualities,frac,random_state): #creating a sample space based on given data; this sample space can be helpful for many situation like fraction-based trainng or keeping track of data or etc. 

        if extractingMethod == 'full':
            middle = list(range(1,numOfFrames-1,1))
        elif extractingMethod == 'even':
            middle = [i-1 for i in range(0,numOfFrames,2) if i > 0]
        else:
            raise("not supported yet -> full|even")
        
        if self.train:
            if type(totalQualities) != list:
                qp = list(range(1,totalQualities+1))
            else:
                qp=totalQualities
            combinations = list(itertools.product(seqNumbers,middle,qp))
            df = pd.DataFrame(combinations, columns=['seqNum','middle', 'qp',])
            df = df.sort_values(by=['seqNum','middle', 'qp',]).reset_index(drop=True)
            
        else:
            qp = [totalQualities]
            offsetW = np.arange(0,self.VisibleWindow['width'],self.cropSize)
            offsetH = np.arange(0,self.VisibleWindow['height'],self.cropSize)
            combinations = list(itertools.product(seqNumbers,middle,qp,offsetW,offsetH))
            df = pd.DataFrame(combinations, columns=['seqNum','middle', 'qp',"offsetW","offsetH"])
            df = df.sort_values(by=['seqNum','middle', 'qp',"offsetW","offsetH"]).reset_index(drop=True)
        


        if frac != 1:
            return df.sample(frac=frac,random_state=random_state).reset_index(drop=True)
        else:
            return df

    def _createAugmentions(self,p=.5):
        return ReplayCompose([
        Flip(p),
        RandomRotate90(p),
    ],)

    def _logmapper(self,x):
            if type(x) == int:
                return str(x)
            if type(x) == list:
                x = np.array(x)
                x = x.flatten()
                x = x.tolist()
                return ''.join(map(str,x))
            
    def renewSampleSpace(self,random_state):
        if self.frac != 1:
            self.self.sampleSpace = self._createSampleSpace(self.seqNumbers,self.numOfFramesPerSeq,self.extractingMethod,self.totalQualities,self.frac,random_state)
        else:
            pass

    def _cropANDdownscale(self,xImages,yImages,lookaheadImages,offsetH=None,offsetW=None,):
 
        if offsetH is None and offsetW is None:
            offsetW = random.randint(0,(self.VisibleWindow['width'] - self.cropSize))
            offsetH = random.randint(0,(self.VisibleWindow['height'] - self.cropSize))
        left = offsetW
        right = offsetW + self.cropSize
        top = offsetH
        bottom = offsetH + self.cropSize

        xCropped = xImages.copy()[:,top:bottom,left:right,:]

        if self.augmentation:
            data = self.transfrom(image=xCropped[0])
            xCropped[0] = data['image']
            for i,img in enumerate(xCropped[1:]):
                xCropped[i+1]=ReplayCompose.replay(data['replay'], image=img)['image']

        yCropped = yImages.copy()[:,top:bottom,left:right,:]

        if self.augmentation:
            for i,img in enumerate(yCropped):
                yCropped[i]=ReplayCompose.replay(data['replay'], image=img)['image']

        lookaround= []
        for i,img in enumerate(xImages):
            if i != 1:
                img = cv2.resize(img, (self.cropSize, self.cropSize), interpolation=cv2.INTER_CUBIC)
                if self.augmentation:
                    img=ReplayCompose.replay(data['replay'], image=img)['image']
                lookaround.append(img)

        if len(lookaheadImages.shape) != 4:
            raise Exception("lookahead images shape is incorrect")

        lookaheadCropped=lookaheadImages.copy()[:,top:bottom,left:right,:]
        if self.augmentation:
            for i,img in enumerate(lookaheadCropped):
                lookaheadCropped[i]=ReplayCompose.replay(data['replay'], image=img)['image']

        lookaheadScaled = []
        for img in lookaheadImages:
            img = cv2.resize(img, (self.cropSize, self.cropSize), interpolation=cv2.INTER_CUBIC)
            if self.augmentation:
                img=ReplayCompose.replay(data['replay'], image=img)['image']
            lookaheadScaled.append(img)
        
        scale_x = self.cropSize / self.VisibleWindow['width']
        scale_y = self.cropSize / self.VisibleWindow['height']

        topn = int(top * scale_y)
        bottomn = int(bottom * scale_y)
        leftn = int(left * scale_x)
        rightn = int(right * scale_x)

        lookaround = np.array(lookaround)

        if type(lookaheadScaled) == list:
            lookaheadCropped=np.array(lookaheadCropped)
            lookaheadScaled=np.array(lookaheadScaled)

        if self.augmentation:
            return yCropped,xCropped,lookaheadCropped,lookaheadScaled,lookaround,(left,top),(leftn,topn),data['replay'] 
        else:
            return yCropped,xCropped,lookaheadCropped,lookaheadScaled,lookaround,(left,top),(leftn,topn)

    def _typeCasting(self,xCropped,yCropped,around,aheadCropped,aheadScaled):
        return xCropped.astype(np.float32)/255.,yCropped.astype(np.float32)/255.,around.astype(np.float32)/255.,aheadCropped.astype(np.float32)/255.,aheadScaled.astype(np.float32)/255.

    def _rearrangeImgs(self,xCropped,yCropped,around,aheadCropped,aheadScaled): #rearranging data
        xCropped = rearrange(xCropped, 'f h w c -> c f h w')
        yCropped = rearrange(yCropped, 'f h w c -> c f h w')
        aheadCropped = rearrange(aheadCropped, 'f h w c -> c f h w')
        aheadScaled = rearrange(aheadScaled, 'f h w c -> c f h w')
        around = rearrange(around, 'f h w c -> c f h w')

        return xCropped,yCropped,around,aheadCropped,aheadScaled

    def _logger(self,index,augmented): #for keeping track of data
        
        aug = [[_['__class_fullname__'],''.join(map(self._logmapper,list(_['params'].values())))] for _ in augmented['transforms'] if _['applied'] == True]
        aug = np.array(aug)
        titles = [_['__class_fullname__'] for _ in augmented['transforms']]

        res = []
        for title in titles :
            idx = np.where(aug == title)
            if idx[0].size !=0:
                if title == 'InvertImg':
                    res.append('Applied')
                else:
                    res.append(aug[idx[0],1][0].replace('512',''))
            else:
                res.append('NotApplied')
        #print(augmented)

        if os.path.isfile("./log_data.csv"):
            with open('./log_data.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(index+res)
        else:
            with open('./log_data.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                field = ["video","qp", "middle","left_org","top_org", "left","top","left_scaled","top_scaled"]+titles
                writer.writerow(field)
                writer.writerow(index+res)

    def __init__(self, seqNumbers,numOfFramesPerSeq,rawPath,qpPath,extractingMethod,frac,random_state,augmentation,train,totalQualities=100,cropSize=512,height=4320,width=8192,):
        self.VisibleWindow = {'height':4096,'width':7680} #actual 8K resolution, sepe8k has bigger resolution
        self.seqNumbers = seqNumbers #list of consider videos for training
        self.numOfFramesPerSeq = numOfFramesPerSeq #number of frames per clip/video
        self.rawPath = rawPath #path for raw file
        self.qpPath = qpPath #path for encoded-decoded file
        self.widthOrg =  #size of clip/video's width
        self.heightOrg = height #size of clip/video's width
        self.frac = frac #fraction of data to be used for training
        self.cropSize = cropSize
        self.random_state = random_state
        self.extractingMethod = extractingMethod
        self.totalQualities = totalQualities #number of steps/qualities/QP values
        self.train = train #flag if this data to be considered for training or testing/validating
        self.augmentation = augmentation #flag for using augmentation
        self.transfrom = self._createAugmentions()
        if not train:
            offsetW = np.arange(0,self.VisibleWindow['width'],self.cropSize)
            offsetH = np.arange(0,self.VisibleWindow['height'],self.cropSize)
            self.offset = np.array(list(itertools.product(offsetW,offsetH)))
        self.sampleSpace = self._createSampleSpace(seqNumbers,numOfFramesPerSeq,extractingMethod,totalQualities,frac,random_state,)

    def __len__(self):
        return len(self.sampleSpace)
    
    def __getitem__(self, index):
        if self.train:
            seqNum,middle,qp =self.sampleSpace[self.sampleSpace.index==index].values[0]
        else:
            seqNum,middle,qp,offsetW,offsetH =self.sampleSpace[self.sampleSpace.index==index].values[0]

        xImages,yImages,lookaheadImages,frameNos,types,(leftorg,toporg)= self._loadImages([middle-1,middle,middle+1],seqNum,qp)
        
        if self.augmentation:
            if self.train:
                yCropped,xCropped,aheadCropped,aheadScaled,around,(widthSP,lentghSP),(widthSl,lentghSl),augmented = self._cropANDdownscale(xImages,yImages,lookaheadImages)
            else:
                yCropped,xCropped,aheadCropped,aheadScaled,around,(widthSP,lentghSP),(widthSl,lentghSl),augmented = self._cropANDdownscale(xImages,yImages,lookaheadImages,offsetH,offsetW)
        else:
            if self.train:
                yCropped,xCropped,aheadCropped,aheadScaled,around,(widthSP,lentghSP),(widthSl,lentghSl) = self._cropANDdownscale(xImages,yImages,lookaheadImages)
            else:
                yCropped,xCropped,aheadCropped,aheadScaled,around,(widthSP,lentghSP),(widthSl,lentghSl) = self._cropANDdownscale(xImages,yImages,lookaheadImages,offsetH,offsetW)

        xCropped,yCropped,around,aheadCropped,aheadScaled=self._typeCasting(xCropped,yCropped,around,aheadCropped,aheadScaled)
        aheadCropped = np.concatenate([xCropped[-1][np.newaxis,...],aheadCropped],axis=0)
        aheadScaled = np.concatenate([around[-1][np.newaxis,...],aheadScaled],axis=0)
        xCropped,yCropped,around,aheadCropped,aheadScaled = self._rearrangeImgs(xCropped,yCropped,around,aheadCropped,aheadScaled)
        decayFactor=np.array((self.numOfFramesPerSeq-middle)/self.numOfFramesPerSeq).astype(np.float32)
        return xCropped,yCropped,around,aheadCropped,aheadScaled,np.array([[qp//3],[middle//2],[widthSP],[lentghSP],[widthSl],[lentghSl]]), (seqNum,middle,qp), decayFactor
 
 
 
