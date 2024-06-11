import scipy
import numpy as np

NP_DATA_TYPE = np.float64
NP_IDX_TYPE = np.int64

np.set_printoptions(precision=50)

def check_order(mappings) -> bool:
    """检查映射值的次序，要求映射值整数部分单调递增"""
    count = 0
    for i in range(mappings.shape[0] - 1):
        if int(mappings[i]) > int(mappings[i + 1]):
            print(f'error: i = {i}, map[i] = {mappings[i]}, map[i + 1] = {mappings[i + 1]}')
            count += 1
    if count > 0:
        print('$ check_order ERROR_COUNT: ', count)
        return False
    return True
