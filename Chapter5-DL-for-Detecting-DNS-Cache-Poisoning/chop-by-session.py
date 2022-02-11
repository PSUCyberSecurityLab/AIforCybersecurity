import os

import numpy as np
import pyshark
from tqdm import tqdm

import multiprocessing

def process_benign(benign):
    benign_bytes=list()
    i=0
    j=-1
    for item in tqdm(benign,ascii=True,desc="Processing benign data"):
        cap=pyshark.FileCapture(item,display_filter='dns and not tcp',use_json=True,include_raw=True)
        try:
            while(True):
                pkt=cap.next()
                if pkt.ip.src_host=='192.168.100.128' and pkt.ip.dst_host=='192.168.100.50':
                    benign_bytes.append([])
                    j+=1
                benign_bytes[j].append(np.array(list(int(ele) for ele in pkt.get_raw_packet()[14:26]+pkt.get_raw_packet()[34:54])))
                i+=1
        except StopIteration:
            pass
        cap.close()

    np.save(os.path.join('data','benign_bytes.npy'),np.array(benign_bytes,dtype=object),allow_pickle=True)

def process_malicious(malicious):
    malicious_bytes=list()
    i=0
    j=-1
    for item in tqdm(malicious,ascii=True,desc="Processing malicious data"):
        cap=pyshark.FileCapture(item,display_filter='dns and not tcp',use_json=True,include_raw=True)
        try:
            while(True):
                pkt=cap.next()
                if pkt.ip.src_host=='192.168.100.18' and pkt.ip.dst_host=='192.168.100.50':
                    malicious_bytes.append([])
                    j+=1
                malicious_bytes[j].append(np.array(list(int(ele) for ele in pkt.get_raw_packet()[14:26]+pkt.get_raw_packet()[34:54])))
                i+=1
        except StopIteration:
            pass
        cap.close()

    np.save(os.path.join('data','malicious_bytes.npy'),np.array(malicious_bytes,dtype= object),allow_pickle=True)

if __name__=="__main__":
    multiprocessing.freeze_support()
    benign=[
        os.path.join("data","benign-dns_split"+str(i)+".pcapng") for i in range(1,56)
    ]
    malicious=[
        os.path.join("data","malicious-dns_split"+str(i)+".pcapng") for i in range(1,6)
    ]

    ben_proc=multiprocessing.Process(target=process_benign,args=(benign,))
    mal_proc=multiprocessing.Process(target=process_malicious,args=(malicious,))

    ben_proc.start()
    mal_proc.start()

    ben_proc.join()
    mal_proc.join()
