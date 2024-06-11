import numpy as np
import random


data = np.load(r'C:\Users\mzq\Desktop\论文待发\实验\TOM\data\data_0.npy')
data_512 = data[:512]
np.save(r'C:\Users\mzq\Desktop\论文待发\实验\TOM\data\IO_sim_512.npy', data_512)


pass
