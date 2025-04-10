
import os 
import numpy as np
import cv2
from tqdm import tqdm
import csv
import threading
from utils import split_list

def func(dir_path):

    numOfFrames = 300
    Slidingwindow = numOfFrames - 1

    i=int(dir_path.split("/")[-1])
    for j in range(Slidingwindow):
        main_path = os.path.join(dir_path,"{:03d}_8K.png".format(j))
        main = cv2.imread(main_path)
        if main.max() < 2:
            print(main.dtype)
        main = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        for k in range(Slidingwindow-j):
            k = k + j + 1
            if k == Slidingwindow:
                break
            second_path = os.path.join(dir_path,"{:03d}_8K.png".format(k))
            second  = cv2.imread(second_path)
            second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.

            diff = second - main

            mean = diff.mean()
            miin = diff.min()
            maxx = diff.max()
            var = diff.var()
            std = diff.std()
            nonzero = np.count_nonzero(diff)/diff.size

            if os.path.isfile("./data/window.csv"): #this file provide the difference metrics per video-window_size for each frame.
                with open('./data/window.csv', 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow([i,j,k,mean,miin,maxx,var,std,nonzero])
            else:
                with open('./data/window.csv', 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    field = ["video","first", "second","Avg","min", "max","var","std","nonzero"]
                    writer.writerow(field)
                    writer.writerow([i,j,k,mean,miin,maxx,var,std,nonzero])



def main():
    
    Path = "Path to Raw frames"
    seqNumbers = [] 
    Paths = [os.path.join(Path,f"{i+1:03d}") for i in seqNumbers]
    args = split_list(Paths, 40)
    threads = []
    for arg in args:
        thread = threading.Thread(target=func  , args=arg)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()

