import os
import numpy as np
import glob
import config
class RNNSeq(object):
    def __init__(self, sub_wins, label):
        self.num_sub_wins = len(sub_wins) 
        self.label = label                   
        self.sub_wins = sub_wins   

def Seqdata():
    #files = glob.glob('data/binary/Virus/*') + glob.glob('data/binary/Benign/*')
    files=glob.glob("data/CFG_hex/Benign/*")+glob.glob("data/CFG_hex/Virus/*")
    os.makedirs(os.path.dirname('data/Seq-ori/Virus/'), exist_ok=True)
    os.makedirs(os.path.dirname('data/Seq-ori/Benign/'), exist_ok=True)
  
    for f in files:
        f ='data/binary/'+'/'.join(f.split('/')[-2:]) 
        tmp = np.fromfile(f, np.uint8)
        tmp = np.array(tmp)
        length = len(tmp) if len(tmp) % config.sequence_length == 0 else config.sequence_length - len(tmp) % config.sequence_length + len(tmp)
        res = np.ones(length) * (config.max_features-1)#np.ones(config.max_size) * (config.max_features-1)
        res[:tmp.shape[0]] = tmp
        res = np.reshape(res, (-1, config.sequence_length))
        np.savetxt('data/Seq-ori/'+f.lstrip('data/binary/')+'.txt', res,  fmt='%d') 
     
if __name__ == '__main__':
    Seqdata()