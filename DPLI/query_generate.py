import numpy as np
import random


def get_range_query_total_random(data_npy: np.ndarray, save_path: str, query_num=100):
    """获取一定数量的范围查询，每个查询的范围完全随机"""
    query_lists = []
    pos_arr = data_npy[:, 0]
    t_arr = data_npy[:, 1]
    pos_min, pos_max = pos_arr.min(), pos_arr.max()
    t_min, t_max = t_arr.min(), t_arr.max()

    for i in range(query_num):
        query_list = []

        pos_1 = random.uniform(pos_min, pos_max)
        pos_2 = random.uniform(pos_min, pos_max)
        t_1 = random.uniform(t_min, t_max)
        t_2 = random.uniform(t_min, t_max)

        pos_list = [pos_1, pos_2]
        t_list = [t_1, t_2]
        pos_list.sort()
        t_list.sort()

        query_list.extend(pos_list)
        query_list.extend(t_list)
        query_lists.append(query_list)

    assert save_path.endswith('.txt')
    with open(save_path, 'w') as writer:
        for query in query_lists:
            pos_min, pos_max, t_min, t_max = query
            writer.write(f'{pos_min},{pos_max},{t_min},{t_max}\n')

    print('$ get_range_query_total_random: DONE.')


if __name__ == '__main__':
    test_dir = r'C:\Users\mzq\Desktop\test'
    data_pth = fr'{test_dir}\chd_pos_t_0.npy'

    save_pth = r'C:\Users\mzq\Desktop\TOM_range_query\chd_query_0.txt'

    data_npy = np.load(data_pth)
    get_range_query_total_random(data_npy, save_pth)
    pass
