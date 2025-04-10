import os
import subprocess
import timeit
import threading
import random
from utils import split_list

def decode(*videos):
    i = 1
    cnt = len(videos)
    for root,f_name in videos:
        start_time = timeit.default_timer()
        video = os.path.join(root,f_name)
        output_pattern = os.path.join(root, "%03d.png")
        ffmpeg_command = [
        '/usr/bin/ffmpeg',
        '-hwaccel', 'cuda',        
        '-c:v', 'av1_cuvid', #or hevc_cuvid
        '-i', video,
        '-r', '30', 
        output_pattern]
        subprocess.run(ffmpeg_command)

        end_time = timeit.default_timer()

        with open('decode.log', 'w') as f:
            f.write('video {0} out of {1} is done. {2} seconds taken and {3} minutes time expecting'\
                .format(str(i),str(cnt),str((end_time-start_time)),\
                     str(((end_time-start_time)/60)*(cnt-i))))
        i=i+1

def main():
    VIDEOS_ROOT_PATH = "Path To Videos' root directroy"
    videosToDecode= []
    for root,_,f_names in os.walk(VIDEOS_ROOT_PATH):
        if f_names:
            if '.mp4' in f_names[0]:
                videosToDecode.append([root,f_names[0]])

    args = split_list(videosToDecode, 8)

    threads = []
    for arg in args:
        thread = threading.Thread(target=decode  , args=arg)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
