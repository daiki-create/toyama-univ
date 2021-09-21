import numpy as np

def get_error(t,y,batch_size):
    return -np.sum(t*np.log(y+1e-7))/batch_size


#出力

#正解

error=get_error(,,1)