import nump as np
import itertools
import pandas as pd

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def createSampleSpace(seqNumbers,numOfFrames,extractingMethod,totalQualities,frac,random_state):

        if extractingMethod == 'full':
            middle = list(range(1,numOfFrames-1,1))
        elif extractingMethod == 'even':
            middle = [i-1 for i in range(0,numOfFrames,2) if i > 0]
        else:
            raise("not supported yet -> full|even")
        
        qp=totalQualities

        combinations = list(itertools.product(seqNumbers,middle,qp))
        df = pd.DataFrame(combinations, columns=['seqNum','middle', 'qp'])
        df = df.sort_values(by=['seqNum','middle', 'qp']).reset_index(drop=True)
        
        if frac != 1:
            return df.sample(frac=frac,random_state=random_state).reset_index(drop=True)
        else:
            return df