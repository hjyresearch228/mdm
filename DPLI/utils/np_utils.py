import numpy as np
import sys

INT = np.int64
FlOAT = np.float64

# sys.path.append('.../.../')

def convert_txt_to_npy(src: str, start_line: int):
    """txt格式：每行数据按 ',' 分隔"""
    data = np.loadtxt(src, dtype=str)
    row = data.shape[0]
    data_new = []
    for i_ in range(start_line, row):
        r_ = data[i_].split(',')
        item = [float(_) for _ in r_]
        # data_new.append(item)
        data_new.append([item[0], item[1]])  # 可修改

    dst = src.replace('.txt', '')
    np.save(f'{dst}.npy', data_new)


def convert_npy_to_txt(src: str, dst: str):
    """
    将.npy文件转换为.txt文件
    :param src: .npy文件路径
    :param dst: .txt文件路径
    """
    print('$ convert_npy_to_txt')

    data = np.load(src)
    print(f'data.shape = {data.shape}')
    shape_len = len(data.shape)
    data = data.tolist()

    with open(dst, 'w') as writer:
        if shape_len == 1:
            for x in data:
                writer.write('%.10f\n' % x)
        else:
            for row_data in data:
                writer.write(' '.join(['%.10f' % i for i in row_data]) + '\n')

    print('$$ convert_npy_to_txt: done')