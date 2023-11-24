import matplotlib.pyplot as plt
import numpy as np

raw = '/home/huy/nlp/NMT-LaVi/data/raw/'
file_list_lo = ['dev2023.lo',
                'train2023.lo']
file_list_vi = ['dev2023.vi',
                'train2023.vi']
file_list = file_list_lo + file_list_vi
dup = '/home/huy/nlp/NMT-LaVi/data/dup/'
pre_processed = '/home/huy/nlp/NMT-LaVi/data/pre_processed/'


for file in file_list:
    with open(pre_processed + file,'r') as f:
        print(file)
        line_len = dict()
        lines = f.readlines()
        for line in lines:
            if line_len.get(len(line)) == None:
                line_len[len(line)] = 1
            line_len[len(line)] += 1
        
        
        plt.hist(line_len,10)
        plt.show()