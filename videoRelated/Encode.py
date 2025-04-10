import os
import ffmpeg
import numpy as np
import timeit
import itertools
import threading
from utils import split_list


ORG_SEQ = "Path to Raw frames"
TO_SAVE = "Path to save encoded videos"

def encode(frameDir,sequenceNum,qp,i,cnt):

    _dir = os.path.join("{:03d}".format(sequenceNum),"QP-{}".format(qp))

    os.makedirs(os.path.join(TO_SAVE,"{:03d}".format(sequenceNum)),exist_ok=True)
    os.makedirs(os.path.join(TO_SAVE,_dir),exist_ok=True)


    input_pattern = os.path.join(frameDir, "%03d_8K.png",)
    output_video = os.path.join(os.path.join(TO_SAVE,_dir), "qp_{:03d}.mp4".format(qp),)


    start_time = timeit.default_timer()

    ffmpeg.input(
        input_pattern, 
        framerate=30,
    ).output(
        output_video,
        codec='av1_nvenc', #or hevc_nvenc
        qp=qp,
    ).run(cmd= "/usr/bin/ffmpeg")

    end_time = timeit.default_timer()

    with open('encode.log', 'w') as f:
        f.write('video {0} out of {1} is done. {2} seconds taken and {3} minutes time expecting'\
                .format(str(i),str(cnt),str((end_time-start_time)),\
                     str(((end_time-start_time)/60)*(cnt-i))))


    return _dir,output_video

def groupEncode(*args):
    i = 1
    for (frameDir,sequenceNum),qp in args:
        encode(frameDir,sequenceNum,qp,i,len(args))
        i = i+1

def main():
    qpValues = [] #list of QPs value
    framesDir = [[x[0],i] for i,x in enumerate(os.walk(ORG_SEQ)) if i>0]
    combinations = list(itertools.product(framesDir,qpValues))
    args = split_list(combinations, 8)
    threads = []

    for arg in args:
        thread = threading.Thread(target=groupEncode  , args=arg)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
