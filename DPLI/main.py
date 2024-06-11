import numpy as np

from utils.pkl_utils import *

from models.DPLI import DPLI

# from model_config import *

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
    test_dir = r'C:\Users\mzq\Desktop\test'
    data_pth = fr'{test_dir}\chd_pos_t_0.npy'
    # data_pth = fr'{test_dir}\chd_pos_t_1.npy'
    # data_pth = r'C:\Users\mzq\Desktop\PPT_data\period_0_rf.npy'

    # 获取正确查询结果
    # low_bd = np.array([8184.3872362, 1407059037.3751235])
    # high_bd = np.array([8193.1666134, 1407059525.2817743])
    # qr_1 = np.array([low_bd, high_bd])
    # res = get_right_query_result(data_pth, qr_1)

    data_npy = np.load(data_pth)

    my_index = DPLI('chd', 100, 100, 0.5, 500)
    # my_index = TOM()

    print('------------------------------------------load-------------------------------------------')
    # my_index.load(1)

    print('------------------------------------------bulid-------------------------------------------')
    my_index.initialize(data_npy, 0)

    print('------------------------------------------query-------------------------------------------')
    # real number 417, IO 168
    # low_bd = np.array([3490.747408367292, 1407063486.555564])  # 6:00 - 1407016800  12:00 - 1407124800  24:00 - 1407168000
    # high_bd = np.array([3788.1863160916196, 1407063783.9944715])

    # real number 336
    # low_bd = np.array([100, 1407063486])  # 7
    # high_bd = np.array([200, 1407063783])

    # real number 106
    # low_bd = np.array([100, 1407063486])  # 0
    # high_bd = np.array([125, 1407063786])

    # real number 28
    low_bd = np.array([100, 1407063486])  # 0
    high_bd = np.array([105, 1407063786])

    qr_1 = np.array([low_bd, high_bd])
    # low_bd = np.array([102, 1407103200])
    # high_bd = np.array([103, 1407168000])
    # qr_2 = np.array([low_bd, high_bd])
    qr_list = [qr_1]
    for qr in qr_list:
        qry_data, IO, t_use = my_index.range_query(qr, 0)
        # print(f'query result: {qry_data.shape[0]} \n {qry_data}')
        print(f'query result: {qry_data.shape[0]}')
        print(f'query I/O: {IO}')

    print('------------------------------------------insert-------------------------------------------')
    # data_pth_2 = fr'{test_dir}\chd_pos_t_1.npy'
    # data_npy_2 = np.load(data_pth_2)
    # my_index.bulk_insert(data_npy_2, 1)

    print('------------------------------------------save-------------------------------------------')
    # my_index.save()

    pass
