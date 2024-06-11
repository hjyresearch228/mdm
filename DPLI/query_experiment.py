"""查询实验：I/O次数，时间"""
import random
import os
import numpy as np
import pandas as pd

from TOM.models.TOM import TOM


# from TOM.utils.core.model_config import *


def generate_query_data(dir: str, query_file):
    file_list = os.listdir(dir)
    for file_ in file_list:
        ff = os.path.splitext(file_)[0]
        query_path = query_file + "\\" + ff + ".txt"
        with open(query_path, 'w') as f:
            pass
        file_path = os.path.join(dir, file_)

        data_df = pd.read_csv(file_path, delimiter=',', header=None, names=['pos', 't'])
        data_df = data_df[1:]
        max_pos = data_df['pos'].astype(float).max()
        min_pos = data_df['pos'].astype(float).min()
        max_t = data_df['t'].astype(float).max()
        min_t = data_df['t'].astype(float).min()
        area = (max_t - min_t) * (max_pos - min_pos)
        print(area)
        list_ = [0.00001, 0.00004, 0.00008, 0.00016]
        with open(query_path, 'a') as fff:
            for l_ in list_:
                area_part = area * l_
                print(area_part)
                for i in range(25):
                    lt = random.uniform(min_t, max_t - area_part ** 0.5)
                    lp = random.uniform(min_pos, max_pos - area_part ** 0.5)
                    # 计算右上角坐标
                    rt = lt + area_part ** 0.5
                    rp = lp + area_part ** 0.5
                    fff.write(f'{lp},{rp},{lt},{rt}\n')


def get_queries_for_one_pd(src: str) -> list:
    """获取单期数据的查询序列，参数为单期数据的查询框文件"""
    query_list = []
    with open(src, 'r') as reader:
        query_lines = reader.readlines()
    for i, query_line in enumerate(query_lines):
        query_str_list = query_line.replace('\n', '').split(',')
        query_list_ = [float(str_) for str_ in query_str_list]

        pos_btm, pos_top, t_btm, t_top = query_list_  # tom

        # t_btm, pos_btm, t_top, pos_top = query_list_  # ppt

        # t, pos = query_list_  # ppt point query
        # pos_btm, t_btm = pos, t
        # pos_top, t_top = pos, t

        btm = np.array([pos_btm, t_btm])
        top = np.array([pos_top, t_top])
        qr_ = np.array([btm, top])
        query_list.append(qr_)

    # random.shuffle(query_list)
    return query_list


def get_right_query_result(data_path: str, query_range: np.ndarray):
    """正确的查询结果，返回数据个数"""
    data = np.load(data_path)
    low_bd, high_bd = query_range
    pos_low, t_low = low_bd
    pos_high, t_high = high_bd

    pos_arr = data[:, 0]
    idx_ = np.argwhere((pos_arr <= pos_high) & (pos_arr >= pos_low)).flatten()
    data_ = data[idx_]
    t_arr = data_[:, 1]
    idx_ = np.argwhere((t_arr <= t_high) & (t_arr >= t_low)).flatten()

    data_res = data_[idx_]
    return len(idx_)


if __name__ == '__main__':
    # data_src = r'C:\Users\mzq\Desktop\my_LISA\test\test_300w.txt'
    # data_npy = r'D:\BaiduNetdiskDownload\chd_data\chd_pos_t_1_refined.npy'

    _ = r'C:\Users\mzq\Desktop\新建文件夹'
    # generate_query_data(_, query_src)

    # query_src = r'C:\Users\mzq\Desktop\TOM_range_query\chd_query_0.txt'
    query_src = r'C:\Users\mzq\Desktop\TOM_range_query\chd_pos_t_0_refined.txt'
    # query_src = r'C:\Users\mzq\Desktop\旧\PPT对比实验\chengdu_range_query\period_0.txt'

    test_dir = r'C:\Users\mzq\Desktop\test'
    data_pth = fr'{test_dir}\chd_pos_t_0.npy'
    # data_pth = r'C:\Users\mzq\Desktop\PPT_data\period_0_rf.npy'
    data_npy = np.load(data_pth)

    qr_list = get_queries_for_one_pd(query_src)

    # low_bd = np.array([1501, 1407103200])  # 6:00 - 1407016800  12:00 - 1407124800  24:00 - 1407168000
    # high_bd = np.array([1502, 1407168000])
    # qr_1 = np.array([low_bd, high_bd])
    # low_bd = np.array([1502, 1407103200])
    # high_bd = np.array([1503, 1407168000])
    # qr_2 = np.array([low_bd, high_bd])
    # qr_list = [qr_1, qr_2]

    my_index = TOM('chd', 100, 100)
    my_index.initialize(data_npy, 0)
    # my_index.load(1)

    # debug
    low_ = [0, 1407124800]
    high_ = [100, 1407125100]
    qr_ = [low_, high_]
    real_num = get_right_query_result(data_pth, qr_)

    IO_list = []
    time_list = []
    recall_list = []
    result_num = 0
    query_num = 0
    for i, qr in enumerate(qr_list):
        low, high = qr
        pos, t = low
        pos_, t_ = high
        if pos > 19750 or pos_ > 19750:
            continue

        qry_data, IO, t_use = my_index.range_query(qr, 0)
        # print(f'query result: {qry_data.shape[0]} \n {qry_data}')
        # print(f'query result: {qry_data.shape[0]}')
        # print(f'query I/O: {IO}')
        cur_num = qry_data.shape[0]
        real_num = get_right_query_result(data_pth, qr)
        if real_num == 0:
            if cur_num == real_num:
                recall = 1
            else:
                recall = 0
        else:
            recall = cur_num / real_num

        if recall < 1:
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            print(f'real num {real_num}, cur num {cur_num}')
            print(f'^^^^^^ pos_low: {round(float(pos), 7)}, t_low: {round(float(t), 7)}')
            print(f'^^^^^^ pos_high: {round(float(pos_), 7)}, t_high: {round(float(t_), 7)}')
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        result_num += real_num
        query_num += cur_num
        recall_list.append(recall)
        IO_list.append(IO)
        time_list.append(t_use)

    avg_IO = round(np.mean(IO_list), 3)
    avg_time = round(np.mean(time_list), 3)
    avg_recall = round(np.mean(recall_list), 3)

    print(f'average I/O: {avg_IO}')
    print(f'average time: {avg_time} s')
    print(f'average recall: {avg_recall}')
    print(f'query result number: {query_num}')
    print(f'real data number: {result_num}')

    pass
