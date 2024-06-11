import copy
import math
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# from src.utils.np_utils import *

Line = namedtuple('Line', ['slope', 'intercept'])


class DouglasPeucker:
    data = None  # 数据
    data_to_fit = None

    threshold = None  # DP算法阈值
    err_bound = None  # 最大误差
    line_num = None  # 折线段数

    lines = []  # 线段列表[(斜率, 截距)]
    t_bounds = []  # 线段端点列表，两个端点之间由一条线段拟合
    err_bounds = []  # 线段最大误差列表，初始化为threshold
    break_idxs = []  # 分割点索引列表（包含端点）

    def __init__(self, threshold=None):
        self.threshold = threshold
        self.err_bound = threshold
        self.lines = []
        self.t_bounds = []

    def fit(self, data_sorted: np.ndarray):
        """使用DP算法拟合折线段"""
        if len(data_sorted) < 2:
            x, y = data_sorted[0]
            row_ = np.array([x + 1, y + 1]).reshape(1, 2)
            data_sorted = np.append(data_sorted, row_, axis=0)

        self.data = data_sorted

        start = 0
        end = data_sorted.shape[0] - 1
        brk_idxs = []  # 分割点索引序列

        self._DP(start, end, data_sorted, brk_idxs)

        brk_idxs.sort()
        brk_idxs.insert(0, 0)
        brk_idxs.append(end)

        self.break_idxs = brk_idxs

        breaks = data_sorted[brk_idxs]  # 获取分段线段的端点

        t_bds = []
        lines = []
        for from_pt, to_pt in zip(breaks[:-1], breaks[1:]):
            t_0, z_0 = from_pt
            t_1, z_1 = to_pt
            A = (t_0 - t_1) / (z_0 - z_1)
            B = z_1 - t_1 / A
            line_ = (A, B)

            lines.append(line_)
            t_bds.append(t_1)

        # t_bds.insert(0, 0)

        self.t_bounds = t_bds
        self.lines = lines

        self.err_bounds = self._cal_error_bds_for_lines(brk_idxs)
        self.line_num = len(self.lines)
        self.err_bound = int(max(self.err_bounds))
        pass

    def predict(self, t):
        """根据时间戳预测位置"""
        t_bds = self.t_bounds
        t_bds_iter = copy.copy(t_bds)
        t_bds_iter.insert(0, 0)
        idx = 0
        for i_ in range(len(t_bds_iter) - 1):
            if t_bds_iter[i_] < t <= t_bds_iter[i_ + 1]:
                idx = i_
                break
        line_tuple = self.lines[idx]
        A, B = line_tuple
        line = Line(A, B)
        rd = self._predict_rd(t, line)

        return math.ceil(rd)

    def _cal_current_line(self, data_in_line: np.ndarray, line: Line):
        A, B = line
        t_arr = data_in_line[:, 0]
        rd_arr = data_in_line[:, 1]
        arr_ = rd_arr - t_arr / A
        B_cur = np.mean(arr_)
        return B_cur

    def _get_data_in_line(self, data: np.ndarray, t_start, t_end):
        t_arr = data[:, 0]
        idxs_ = np.argwhere((t_arr < t_end) & (t_arr >= t_start)).flatten()
        data_res = data[idxs_]
        return data_res

    def _DP(self, start: int, end: int, data: np.ndarray, splits):
        if start == end:
            return

        t_st, z_st = data[start]
        t_ed, z_ed = data[end]

        A = (t_ed - t_st) / (z_ed - z_st)
        B = z_ed - t_ed / A
        line = Line(A, B)  # 线段line = [斜率, 截距]

        part_data = data[start: end + 1, :]
        split_idx = self._cal_split(part_data, line)

        if not split_idx:  # 没有分割点，返回
            return
        else:
            split_idx_global = int(split_idx + start)
            splits.append(split_idx_global)

        self._DP(start, split_idx_global, data, splits)
        self._DP(split_idx_global, end, data, splits)

    def _cal_split(self, points, line: Line):
        """计算分割点索引"""
        sp = None
        ds = []
        max_d = 0
        for i, pt in enumerate(points):
            dist_ = self._cal_distance(pt, line)
            ds.append(dist_)
            if dist_ > max_d:
                max_d = dist_
                if dist_ > self.threshold:
                    sp = i
        return sp

    @staticmethod
    def _cal_distance(point: np.ndarray, line: Line) -> float:
        """计算预测误差"""
        A = line.slope
        B = line.intercept
        t, z = point
        z_pred = t / A + B  # 计算预测值z
        dist = round(abs(z - z_pred), 5)
        return dist

    def _cal_error_bds_for_lines(self, splits: list):
        err_bds = []
        lines = self.lines
        # 对每条直线求最大误差
        for start, end, line in zip(splits[:-1], splits[1:], lines):
            part_data = self.data[start: end]
            # part_data_ = self.data_to_fit[start: end]
            err_ = self._cal_error_bd_for_line(part_data, line)
            # if err_ > 51:
            #     self._show_line_result(part_data, part_data_, err_, show_order=True)
            err_bds.append(err_)

        return err_bds

    def _cal_error_bd_for_line(self, part_data, line) -> float:
        err_mx = 0
        for data_pt in part_data:
            t, rd = data_pt
            rd_real = rd
            rd_pred = self._predict_rd(t, line)
            err = abs(rd_pred - rd_real)
            err_mx = max(err, err_mx)
        return err_mx

    def _predict_rd(self, t, line):
        """根据t预测rd"""
        A, B = line
        rd_pred = t / A + B
        return rd_pred

    def _generate_data_2d(self, data):
        """生成2维点数据"""
        data = data.reshape(-1, 1)
        data_sorted = self._sort_data(data)
        row = data_sorted.shape[0]
        z_data = np.array([i for i in range(row)]).reshape(row, 1)
        data_2d_sorted = np.append(data_sorted, z_data, axis=1)
        return data_2d_sorted

    def _generate_data_2d_page(self, data):
        """生成2维点数据"""
        data = data.reshape(-1, 1)
        data_sorted = self._sort_data(data)
        row = data_sorted.shape[0]
        z_data = np.array([i for i in range(row)]).reshape(row, 1)
        data_2d_sorted = np.append(data_sorted, z_data, axis=1)
        return data_2d_sorted

    def _sort_data(self, data):
        """将数据按照一定次序排序"""
        measures = []
        for pt_ in data:
            mea_ = pt_[0]
            measures.append(mea_)
        idxs = np.argsort(np.array(measures))
        data_sort = data[idxs]
        return data_sort

    def show_dp_result(self, show_order=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        data = self.data
        splits = self.break_idxs

        split_points = []
        for split_idx in splits:
            pt_ = list(data[split_idx])
            split_points.append(pt_)
        split_points = np.array(split_points)

        sp_xs = split_points[:, 0]
        sp_ys = split_points[:, 1]

        xs = data[:, 0]
        ys = data[:, 1]

        ax.plot(sp_xs, sp_ys, marker='o', alpha=1)
        ax.plot(xs, ys, marker='o', alpha=0.5, ls='')

        if show_order:
            for i in range(len(xs)):
                ax.text(xs[i], ys[i], i)

        ax.set_xlabel('t_range')
        ax.set_ylabel('z')
        plt.title(f'DP result')
        plt.show()

    def _show_line_result(self, data: np.ndarray, data_to_fit: np.ndarray, error, show_order=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        splits = [0, len(data_to_fit) - 1]

        split_points = []
        for split_idx in splits:
            pt_ = list(data_to_fit[split_idx])
            split_points.append(pt_)
        split_points = np.array(split_points)

        sp_xs = split_points[:, 0]
        sp_ys = split_points[:, 1]

        xs = data[:, 0]
        ys = data[:, 1]

        xs_ = data_to_fit[:, 0]
        ys_ = data_to_fit[:, 1]

        ax.plot(sp_xs, sp_ys, marker='o', alpha=1)
        ax.plot(xs, ys, marker='o', alpha=0.5, ls='')
        ax.plot(xs_, ys_, marker='^', alpha=0.5, ls='')

        if show_order:
            for i in range(len(xs)):
                ax.text(xs[i], ys[i], i)

        ax.set_xlabel('t_range')
        ax.set_ylabel('z')
        plt.title(f'max error: {error}')
        plt.show()
