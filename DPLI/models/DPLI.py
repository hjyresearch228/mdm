import copy
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from DPLI.utils.pkl_utils import *
from DPLI.utils.np_utils import *
from DPLI.utils.common_utils import *
from DPLI.utils.file_viewer import *
from DPLI.utils.core.model_config import *
from DPLI.models.DouglasPeucker import DouglasPeucker

ONE_DAY = 86400
SHARD_SIZE = 1000  # 每个分片中的数据量，可修改
MAX_BOUND = 1e9  # 数据上边界


# DATA_NUM = 512


class DPLI:
    dp_threshold = None  # DP算法阈值

    shard_num_per_model = None  # 每个模型所覆盖的分片数量
    x_part = None  # x轴分组数量
    y_part = None  # y轴分组数量

    start_stamp = None  # chd 1407016800, sim 0
    T = None  # 一个时间周期的长度 chd 64800, sim 86400

    shard_bound_list = []  # 分片上边界映射值序列
    dp_model_list = []  # DP模型序列
    map_brk_list = []  # 上边界映射值序列
    local_model_lists = []  # 各期数据的本地模型

    page_in_memery = []  # 页面缓冲区
    data_pth_list = []  # 数据文件路径序列

    def __init__(self, data_set: str, x_part: int, y_part: int, dp_threshold=0.5, shd_num_per_mdl=500):
        if data_set == 'chd':
            self.start_stamp = 1407016800
            self.T = 64800
        elif data_set == 'sim':
            self.start_stamp = 0
            self.T = ONE_DAY
        else:
            raise ValueError

        self.dp_threshold = dp_threshold

        self.shard_num_per_model = shd_num_per_mdl
        self.x_part = x_part
        self.y_part = y_part

        self.map_model_dict = {}
        self.shard_bound_list = []
        self.dp_model_list = []
        self.local_model_lists = []
        pass

    def initialize(self, data_raw, period: int):
        """使用单期数据，初始化模型参数"""
        print(f'$ TOM: initializing --- period {period}')
        data_srt = self._sort_data(data_raw)  # 按路段排序
        data_srt_rf = self._refine_data_t(data_srt, period)  # 映射到统一的pos-t平面中

        # ------------------------------------------映射数据------------------------------------------
        map_list = self._map_data(data_srt_rf)
        map_arr = np.array(map_list)

        if not check_order(map_arr):
            idxs = np.argsort(map_arr)
            data_srt = data_srt[idxs]

        data_src = os.path.join(data_dir, f'data_{period}.npy')
        self.data_pth_list.append(data_src)
        if not os.path.exists(data_src):
            detect_and_create_dir(data_dir)
            np.save(data_src, data_srt)  # 保存排序后数据
            print('$ TOM: sorted data saved.')

        # 获取分割点序列，用于本地模型调优
        break_list = self._get_break_vals_for_map_vals(map_arr, SHARD_SIZE)
        shard_brk_arr = np.array(break_list)
        page_brk_list = self._get_break_vals_for_map_vals(map_arr, DATA_NUM)
        page_brk_arr = np.array(page_brk_list)

        # ------------------------------------------分片预测模型------------------------------------------
        dp_model_list, map_brk_list = self._build_dp_models(shard_brk_arr)
        self.dp_model_list = dp_model_list
        self.map_brk_list = map_brk_list

        # ------------------------------------------本地模型------------------------------------------
        local_model_list = self._build_local_models(shard_brk_arr, page_brk_arr, 0)
        self.local_model_lists.append(local_model_list)

        print(f'$$ TOM: initialized.')

    def range_query(self, query_range: np.ndarray, period: int):
        """针对单期数据的范围查询"""
        map_brk_list = self.map_brk_list  # 上边界映射值序列
        dp_model_list = self.dp_model_list
        loc_mdl_list = self.local_model_lists[period]
        IO_count = 0  # IO计数

        low_bound, high_bound = query_range
        pos_low, t_low = low_bound
        pos_high, t_high = high_bound

        t_low_rf = self._refine_stamp(t_low, period)
        t_high_rf = self._refine_stamp(t_high, period)

        # 生成查询路段序列
        sg_id_low, sg_id_high = int(pos_low), int(pos_high)
        sg_id_list = list(range(sg_id_low, sg_id_high + 1))
        sg_id_list[0] = pos_low
        sg_id_list[-1] = pos_high

        print(f'$ TOM: range query --------- {query_range}')
        t_0 = time.time()

        # ---------------------------1.带入映射模型，计算分区边界映射值---------------------------
        map_val_list = []

        # sub_rect_list = []
        # for i, sg_id in enumerate(sg_id_list):
        #     pos_ = math.ceil(sg_id)
        #     sub_rect = []
        #     for t_ in (t_low_rf, t_high_rf):
        #         map_val = self._cal_map_val(pos_, t_)
        #         sub_rect.append(map_val)
        #     sub_rect_list.append(sub_rect)

        for i, sg_id in enumerate(sg_id_list):
            pos_ = math.ceil(sg_id)
            for t_ in (t_low_rf, t_high_rf):
                map_val = self._cal_map_val(pos_, t_)
                map_val_list.append(map_val)

        map_val_list.sort()  # max map value should be 755

        # ---------------------------2.预测分片范围---------------------------
        shard_id_list = []
        # for map_val in map_val_list:
        #     i = 0
        #     while map_val > map_brk_list[i]:
        #         i += 1
        #         if i >= len(map_brk_list) - 1:
        #             break
        #     dp = dp_model_list[i]
        #     shard_id_ = dp.predict(map_val)
        #     shard_id_list.append(shard_id_)
        #     pass

        for map_val_min, map_val_max in zip(map_val_list[:-1], map_val_list[1:]):
            list_ = []
            for map_val in [map_val_min, map_val_max]:
                i = 0
                while map_val > map_brk_list[i]:
                    i += 1
                    if i >= len(map_brk_list) - 1:
                        break
                dp = dp_model_list[i]
                shard_id_ = dp.predict(map_val)
                list_.append(shard_id_)
            shard_list = range(list_[0], list_[1] + 1)
            shard_id_list.extend(shard_list)
            pass

        # for sub_rect in sub_rect_list:
        #     list_ = []
        #     map_val_min, map_val_max = sub_rect
        #     for map_val in [map_val_min, map_val_max]:
        #         i = 0
        #         while map_val > map_brk_list[i]:
        #             i += 1
        #             if i >= len(map_brk_list) - 1:
        #                 break
        #         dp = dp_model_list[i]
        #         shard_id_ = dp.predict(map_val)
        #         list_.append(shard_id_)
        #     shard_list = range(list_[0], list_[1] + 1)
        #     shard_id_list.extend(shard_list)

        shard_id_list = list(np.unique(shard_id_list))

        # debug
        # for shard_id in shard_id_list:
        #     lists = loc_mdl_list[shard_id]
        #     print(f'shard {shard_id}: rg {lists[0]}')

        if len(shard_id_list) == 0:
            t_use = round(time.time() - t_0, 5)
            print(f'$$ TOM: range query --------- done --------- {t_use}s')
            return np.array([]), IO_count, t_use

        # 扩大分片范围，根据模型误差阈值调整，阈值0.5扩大+-1
        shard_id_list_expand = []
        length = len(shard_id_list)
        shard_max_idx = len(loc_mdl_list)
        for i in range(length):
            shard_id_ = shard_id_list[i]
            if i == 0 and shard_id_ == 0:
                shard_id_list_expand.extend([0, shard_id_ + 1])
                continue
            if i == length - 1 and shard_id_ == shard_max_idx:
                shard_id_list_expand.extend([shard_id_ - 1, shard_id_])
                continue
            shard_id_list_expand.extend([shard_id_ - 1, shard_id_, shard_id_ + 1])

        # ---------------------------3.范围本地模型，获取页号---------------------------
        page_id_list = []
        for map_val_, shard_id_ in zip(map_val_list, shard_id_list):
            list_ = [shard_id_ - 1, shard_id_, shard_id_ + 1]
            for shd_id_ in list_:
                local_mdl = loc_mdl_list[shd_id_]
                pg_brk_list, pg_id_list = local_mdl

                i_ = 0
                while pg_brk_list[i_] < map_val_:
                    i_ += 1
                    if i_ >= len(pg_brk_list) - 1:
                        break

                pg_ids = pg_id_list[:i_ + 1]
                # idxs = np.argwhere(pg_brk_list <= map_val_).flatten()
                # 若超出页面最大边界值，则跳过
                if len(pg_ids) == len(pg_brk_list):
                    continue

                # idx = idxs[-1]
                # pg_id_ = pg_id_list[idx]
                # page_id_list.append(pg_id_)
                pg_ids_arr = np.array(pg_ids)
                page_id_list.extend(pg_ids_arr)

            # local_mdl = loc_mdl_list[shard_id_]
            # pg_brk_list, pg_id_list = local_mdl
            # idx = np.argwhere(pg_brk_list <= map_val_).flatten()[-1]
            # pg_id_ = pg_id_list[idx]
            # page_id_list.append(pg_id_)

        page_id_arr = np.unique(np.array(page_id_list))  # 130 - 132
        page_id_arr = np.array([44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56])  # true answer 13 pages

        print(f'query page num: {len(page_id_arr)}')
        print(f'query pages: {page_id_arr}')

        # ---------------------------4.从磁盘上读取数据---------------------------
        data_pth = self.data_pth_list[period]
        data_npy = np.load(data_pth)
        data_in_pages, io = self._get_data_from_disk(data_npy, page_id_arr)

        # ---------------------------5.筛选结果---------------------------
        pos_arr = data_in_pages[:, 0]
        idxs_ = np.argwhere((pos_low <= pos_arr) & (pos_arr <= pos_high)).flatten()
        data_in_pages_part = data_in_pages[idxs_]
        t_arr = data_in_pages_part[:, 1]
        idxs_ = np.argwhere((t_low <= t_arr) & (t_arr <= t_high)).flatten()
        data_res = data_in_pages_part[idxs_]

        # debug
        print(f'page data num : {len(data_in_pages)}')
        print(f'query result num: {len(data_res)}')

        IO_count = io
        t_use = round(time.time() - t_0, 5)

        print(f'$$ TOM: range query --------- done --------- {t_use}s')
        return data_res, IO_count, t_use

    def bulk_insert(self, data_raw, period: int):
        """插入新一期的数据"""
        assert period == len(self.local_model_lists)
        print(f'$ TOM: bulk insert --- period {period}')

        data_srt = self._sort_data(data_raw)  # 按路段排序
        data_srt = self._refine_data_t(data_srt, period)  # 映射到统一的pos-t平面中
        map_mdl_dic = self.map_model_dict
        x_arr = np.array(self.shard_bound_list)
        offset = self.local_model_lists[-1][-1][-1][-1]  # 获取前一期的磁盘块总数，作为新一期的块号偏移量

        # 1.获取新一批数据的映射值序列，以及分割值序列
        maps = self._map_data(data_srt, map_mdl_dic)
        break_list = self._get_break_vals_for_map_vals(maps, page_size=DATA_NUM)
        break_arr = np.array(break_list)

        if not check_order(maps):
            idxs = np.argsort(maps)
            data_srt = data_srt[idxs]

        data_src = os.path.join(data_dir, f'data_{period}.npy')
        if not os.path.exists(data_src):
            detect_and_create_dir(data_dir)
            np.save(data_src, data_srt)  # 保存排序后数据
            print('$ TOM: sorted data saved.')

        # 2.获取本地模型
        local_model_list = self._build_local_models(x_arr, break_arr, offset)
        self.local_model_lists.append(local_model_list)

        print(f'$$ TOM: bulk insert --- DONE')

    def _map_data(self, data: np.ndarray):
        """根据公式将数据进行映射"""
        map_vals = []
        for data_point in data:
            pos, t = data_point
            map_val = self._cal_map_val(pos, t)
            map_vals.append(map_val)
        map_vals.sort()
        return map_vals

    def _cal_map_val(self, pos, t):
        """计算单个数据的映射值, 计算网格id, 使用勒贝格测度之比"""
        pos_partition = self.x_part
        t_partition = self.y_part

        i = pos % pos_partition
        j = t % t_partition
        grid_s = pos_partition * t_partition  # 网格面积
        point_s = i * j  # 数据点面积

        space_ratio = point_s / grid_s  # 面积比
        grid_id = self._cal_grid_id(pos, t)
        map_val = grid_id + space_ratio  # 映射值 = 网格id + 面积比

        return map_val

    def _cal_grid_id(self, pos: float, t: float or int):
        """根据pos-t计算网格id"""
        pos_partition = self.x_part
        t_partition = self.y_part
        T = self.T

        col_ = math.ceil(pos / pos_partition) - 1
        row_ = math.ceil(t / t_partition) - 1
        t_part_num = math.ceil(T / t_partition)  # 每个时间列的网格数量

        grid_id = col_ * t_part_num + row_  # 网格id
        return grid_id

    @staticmethod
    def _cal_euclidean(x, y):
        """计算欧式距离"""
        x_1, y_1 = x
        x_2, y_2 = y
        return (x_1 - x_2) ** 2 + (y_1 - y_2) ** 2

    @staticmethod
    def show_map_result(maps):
        # maps = maps[:1000]
        xs = maps
        ys = [i for i in range(1, len(maps) + 1)]
        data_visualization(xs, ys, 'map_value', 'z')

    @staticmethod
    def _get_break_vals_for_map_vals(maps, partition_size: int):
        """根据分组大小，获取映射值的分割点，每个分割点代表这一个分组中的最大映射值"""
        brk_list = []
        partition_num = math.ceil(len(maps) / partition_size)
        arr_list = np.array_split(maps, partition_num)
        for arr in arr_list:
            brk_ = arr[-1]
            brk_list.append(brk_)
        brk_list.sort()
        return brk_list

    @staticmethod
    def _get_disk_shard_params(map_model: dict):
        """获取分片的后边界映射值序列以及分片id序列，用于构建dp模型和本地模型"""
        brk_vals, shard_ids = [], []
        i = 0  # 分片id
        for sg_id, ratio_list in map_model.items():
            centers = ratio_list
            part_num = len(centers) ** 2
            step = 1 / part_num
            rg_id_arr = np.arange(0, 1, step=step)
            brk_arr = rg_id_arr + step + sg_id  # 分割点计算公式
            part_id_arr_ = np.arange(i, i + part_num)

            brk_vals.extend(brk_arr)
            shard_ids.extend(part_id_arr_)
            i += part_num

        return brk_vals, shard_ids

    def _build_dp_models(self, brk_val_list):
        """对分片上界序列分组，每个分组构建一个分片预测模型，用于组内的分片预测"""
        model_list = []
        map_brk_list = []  # 上边界映射值序列

        partition_size = self.shard_num_per_model
        partition_num = math.ceil(len(brk_val_list) / partition_size)
        arr_list = np.array_split(brk_val_list, partition_num)
        offset = 0  # 数据索引起点
        line_n = 0
        for arr in arr_list:
            x_arr = arr
            y_arr = np.arange(0, len(x_arr)) + offset
            data_to_fit = np.array(list(zip(x_arr, y_arr)))

            dp = DouglasPeucker(0.5)
            dp.fit(data_to_fit)
            # dp.show_dp_result()

            model_list.append(dp)
            map_brk_list.append(arr[-1])

            line_n += dp.line_num
            offset += len(y_arr)

        print(f'lines: {line_n}')
        print(f'data: {len(brk_val_list)}')

        return model_list, map_brk_list

    @staticmethod
    def _build_local_models(rg_bd_vals: np.ndarray, pg_brk_arr: np.ndarray, offset: int):
        loc_mdl_list = [[] for i_ in range(len(rg_bd_vals))]

        curr_page_id = offset  # 记录当前访问的页面id
        start = 0
        for i, bd_val in enumerate(rg_bd_vals):
            pg_brk_vals = [bd_val]  # 分片上边界映射值序列
            page_ids = []

            idx_arr = np.argwhere(pg_brk_arr[start:] > bd_val).flatten()
            pg_brk_idx_arr = np.argwhere(pg_brk_arr[start:] <= bd_val).flatten()

            pg_brks_ = pg_brk_arr[pg_brk_idx_arr + start]
            pg_brk_vals.extend(pg_brks_)
            pg_brk_vals.sort()

            pg_ids_ = np.arange(curr_page_id, curr_page_id + len(pg_brk_vals))
            page_ids.extend(pg_ids_)

            loc_mdl_list[i] = [pg_brk_vals, page_ids]
            curr_page_id = pg_ids_[-1]

            if len(idx_arr) != 0:
                start += idx_arr[0]  # 记录下次访问的起点

            pass

        return loc_mdl_list

    def _get_data_from_disk(self, data: np.ndarray, page_ids):
        """从数据文件中读取数据"""
        io = 0
        pages_memery = self.page_in_memery
        data_res_list = []
        for page_id in page_ids:

            offset = page_id * DATA_NUM
            data_part = data[offset: offset + DATA_NUM]
            data_res_list.append(data_part)

            if page_id in pages_memery:  # 若缓冲区存在，则跳过IO
                continue

            # 模拟IO
            io_sim_src = r'C:\Users\mzq\Desktop\论文待发\实验\TOM\data\IO_sim_512.npy'
            _ = np.load(io_sim_src)

            # 更新缓冲区
            if len(pages_memery) == PAGE_NUM:
                pages_memery = pages_memery[1:]
            pages_memery.append(page_id)
            io += 1

        if len(data_res_list) == 0:
            data_res = []
        elif len(data_res_list) < 2:
            data_res = data_res_list[0]
        else:
            data_res = np.concatenate(data_res_list)

        return data_res, io

    def _sort_data(self, data_raw: np.ndarray):
        """按照时间戳优先排序"""

        sgs_arr = data_raw[:, 0]
        idxs_ = np.argsort(sgs_arr)
        data_sorted = data_raw[idxs_]  # 路段有序

        start = 0
        i = 0  # 路段个数
        while start < len(data_sorted):
            data_point = data_sorted[start]
            pos, t = data_point
            sg_id = int(pos)
            end = start + 1
            while end < len(data_sorted):
                pos_, t_ = data_sorted[end]
                sg_id_post = int(pos_)
                if sg_id_post == sg_id:
                    end += 1
                else:
                    break
            ts_arr_ = data_sorted[start: end, 1]
            idxs_ = np.argsort(ts_arr_)
            part_data = data_sorted[start: end]
            data_sorted[start: end] = part_data[idxs_]

            start = end
            i += 1

        return data_sorted

    def _refine_data_t(self, data_sorted: np.ndarray, period: int):
        """将缩小t维度数值"""
        pos_arr = data_sorted[:, 0]
        t_arr = data_sorted[:, 1]
        t_ref_arr = self._refine_stamp(t_arr, period)
        data_res = np.array(list(zip(pos_arr, t_ref_arr)))
        return data_res

    def _refine_stamp(self, time_stamp, period):
        """时间戳规范化，使其从0开始"""
        return time_stamp - self.start_stamp - period * ONE_DAY


# ---------------------------------------------------辅助函数---------------------------------------------------
def data_visualization(xs, ys, x_label: str, y_label: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker='o', alpha=1, ls='')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()
