import os

import numpy as np

from utils.pkl_utils import *


def data_filter(data_raw: np.ndarray, sg_ids_common: list):
    """根据公共路段集合过滤数据"""
    pos_arr = data_raw[:, 0]
    idxs_ = np.argsort(pos_arr).flatten()
    data_sort = data_raw[idxs_]
    pos_arr_sort = data_sort[:, 0]

    sg_ids_arr = np.fix(pos_arr_sort)
    sg_ids_uni_arr = np.unique(sg_ids_arr)
    sg_ids_intersec = np.intersect1d(sg_ids_common, sg_ids_uni_arr)  # 交集
    sg_ids_diff = np.setdiff1d(sg_ids_uni_arr, sg_ids_intersec)  # 差集

    idxs = []
    for i, sg_id in enumerate(sg_ids_arr):
        if sg_id in sg_ids_diff:
            continue
        idxs.append(i)

    idxs_arr = np.array(idxs)
    pos_arr_sort_inter = pos_arr_sort[idxs_arr]
    pos_arr_inter_fix = np.fix(pos_arr_sort_inter)
    pos_arr_inter_frac = pos_arr_sort_inter - pos_arr_inter_fix
    data_new = data_sort[idxs_arr]

    for i in range(len(sg_ids_intersec)):
        sg_id = sg_ids_intersec[i]
        idxs_ = np.argwhere(pos_arr_inter_fix == sg_id).flatten()
        pos_arr_inter_fix[idxs_] = i
        pass

    data_new[:, 0] = pos_arr_inter_fix + pos_arr_inter_frac

    return data_new


if __name__ == '__main__':
    test_dir = r'C:\Users\mzq\Desktop\test'
    data_dir = r'D:\BaiduNetdiskDownload\chd_data'
    # data_pth = fr'{test_dir}\chd_pos_t_0.npy'
    # data_pth = fr'{test_dir}\chd_pos_t_1.npy'
    data_pth = r'C:\Users\mzq\Desktop\PPT_data\period_0.npy'
    sg_ids_pth = fr'{test_dir}\sg_ids_common.pkl'

    sg_ids_cm = load_obj(sg_ids_pth)

    data_npy = np.load(data_pth)
    data_npy_512 = data_npy[:512]
    np.save(r'C:\Users\mzq\Desktop\论文待发\实验\TOM\data\IO_sim_512.npy', data_npy_512)

    # data_npy = np.load(data_pth)
    # data_npy_new = data_filter(data_npy, sg_ids_cm)
    # np.save(r'C:\Users\mzq\Desktop\PPT_data\period_0_rf.npy', data_npy_new)

    # files = os.listdir(data_dir)
    # for file in files:
    #     if not file.endswith('.npy'):
    #         continue
    #     data_src = os.path.join(data_dir, file)
    #     data_npy = np.load(data_src)
    #     data_npy_new = data_filter(data_npy, sg_ids_cm)  # 过滤公共路段集合
    #     data_save_src = os.path.join(data_dir, file.replace('refined.npy', 'new.npy'))
    #     np.save(data_save_src, data_npy_new)

    pass
