import math
import os
import random
import struct as st
import time
from collections import defaultdict
from BTree import B_tree
from bisect import bisect_left


class Point:
    # 轨迹点
    # min_time = float('inf')  # 最小时间
    # max_time = -float('inf')  # 最大时间
    # min_x = float('inf')  # 最小经度
    # max_x = -float('inf')  # 最大经度
    # min_y = float('inf')  # 最小纬度
    # max_y = -float('inf')  # 最大纬度
    min_time = 0  # 最小时间
    max_time = 999  # 最大时间
    min_x = 0.0  # 最小经度
    max_x = 50000  # 最大经度
    min_y = 0.0  # 最小纬度
    max_y = 50000  # 最大纬度

    # min_time = 0  # 最小时间
    # max_time = 2678400  # 最大时间
    # min_x = 116.0800006  # 最小经度
    # max_x = 116.7285615  # 最大经度
    # min_y = 39.6800104000001 # 最小纬度
    # max_y = 40.1799936 # 最大纬度
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.time = t
        self.tra_id = -1  # 轨迹id
        self.region_id = None  # 区域id
        self.value = None  # 映射值
        self.order = None  # 序号
        self.predict_value = None  # 预测值

    def __eq__(self, other):
        '''
        判断两个对象是否相同
        :param other: 要比较的对象
        :return: Bool类型
        '''
        return self.x == other.x and self.y == other.y and self.time == other.time

    def __lt__(self, other):
        return self.time < other.time

    def __hash__(self):
        return hash(self.x) * hash(self.y) + hash(self.time)

    def __str__(self):
        return str(self.region_id)


class Linear:
    def __init__(self, x, y, error=0):
        self.x = x  # 映射值
        self.y = y  # 序号
        self.error = error  # 给定误差
        self.model = defaultdict(list)  # 线性模型组[x,y,slope]

    def train(self):
        # print("开始训练模型")
        start_x = self.x[0]
        start_y = self.y[0]
        low_l = -float('inf')
        up_l = float('inf')
        flag = False
        for i in range(1, len(self.x)):
            if self.x[i] == start_x:
                continue
            low = max((self.y[i] - self.error - start_y) / (self.x[i] - start_x), low_l)
            up = min((self.y[i] + self.error - start_y) / (self.x[i] - start_x), up_l)
            if low < up:
                low_l = low
                up_l = up
            else:
                temp = [start_x, start_y, (low_l + up_l) / 2]
                self.model[start_x] = temp
                start_x = self.x[i]
                start_y = self.y[i]
                low_l = -float('inf')
                up_l = float('inf')
                if i == len(self.x) - 1:
                    flag = True
        if flag:
            temp = [start_x, start_y, low_l + random.random()]
        else:
            temp = [start_x, start_y, (low_l + up_l) / 2]
        self.model[start_x] = temp

    def predict(self, test):
        values = list(self.model.keys())
        pre = []
        for i in range(len(test)):
            l, r = 0, len(values) - 1
            while l <= r:
                mid = (l + r) // 2
                if values[mid] == test[i]:
                    r = mid
                    break
                elif values[mid] < test[i]:
                    l = mid + 1
                else:
                    r = mid - 1
            linear = self.model[values[r]]
            y = linear[1] + linear[2] * (test[i] - linear[0])
            pre.append(y)
        return pre


class DataBlock:
    BlockSize = 341
    Block_num = []  # 每个区域存储的块数


class buffer:
    '''
    缓存空间设置
    '''
    InsertTest = []
    InsertId = []
    InsertBlockNum = 0


class File_N:
    path = 'EP1.dat'


def get_Data_From_Trajectory(path):
    '''
    从本地获取轨迹数据，输出为轨迹点的集合
    :param str1: 轨迹的id
    :param trajID: 轨迹的坐标
    :return:
    '''
    file = open(path)
    Point_list = []
    for line in file:
        TraList = line.split(',')
        x = float(TraList[3])
        y = float(TraList[4])
        time = int(TraList[1])
        Traid = int(TraList[0])
        tempPoint = Point(x, y, time)
        tempPoint.tra_id = Traid
        # if x > Point.max_x:
        #     Point.max_x = x
        # if x < Point.min_x:
        #     Point.min_x = x
        # if y > Point.max_y:
        #     Point.max_y = y
        # if y < Point.min_y:
        #     Point.min_y = y
        # if time > Point.max_time:
        #     Point.max_time = time
        # if time < Point.min_time:
        #     Point.min_time = time
        Point_list.append(tempPoint)
    return Point_list


def divide_region(number, points):
    '''
    按照时间划分区域
    :param number: 区域个数
    :param points: 轨迹点
    :return: 时间分割点
    '''
    count = len(points) // number  # 区域中轨迹点的数量
    points.sort()  # 按照时间排序
    time_split = []
    j = count - 1
    for i in range(number - 1):
        if j < len(points):
            while points[j].time == points[j + 1].time:
                j += 1
            time_split.append([j, points[j].time])
            j += count
    r_id = 0
    m = 0
    for n in range(len(points)):
        if m < len(time_split) and n <= time_split[m][0]:
            points[n].region_id = r_id
        elif m < len(time_split) - 1:
            m += 1
            r_id += 1
            points[n].region_id = r_id
        else:
            points[n].region_id = r_id + 1
    return time_split


def lebe_function(row, col, points):
    '''
    映射函数
    :param row: 行数
    :param col: 列数
    :param points: 轨迹点
    :return:
    '''
    inter_x = (Point.max_x - Point.min_x) / col
    inter_y = (Point.max_y - Point.min_y) / row
    area = inter_y * inter_x
    value_collection = defaultdict(list)
    for i in range(len(points)):
        r = (points[i].y - Point.min_y) // inter_y  # 行
        if r >= row:
            r = row - 1
        c = (points[i].x - Point.min_x) // inter_x  # 列
        if c >= col:
            c = col - 1
        cell = r * col + c  # 网格序号
        cell_x = Point.min_x + inter_x * c
        cell_y = Point.min_y + inter_y * r
        if (points[i].y - cell_y) * (points[i].x - cell_x) / area > 1.0:
            print("映射值非法！")
        value = cell + (points[i].y - cell_y) * (points[i].x - cell_x) / area
        points[i].value = value
        value_collection[value].append(points[i])
    # print("数据点数：",len(points))
    # print("映射值个数：",len(value_collection))
    value_collection = sorted(value_collection.items(), key=lambda x: x[0])
    point_order = []
    id = 0
    for tup in value_collection:
        for i in range(len(tup[1])):
            tup[1][i].order = id
            point_order.append(tup[1][i])
        id += 1
    return point_order


def get_NextP_Point(start, end, m):
    '''
    获取某个周期的数据
    :param start: 开始轨迹id
    :param end: 结束轨迹id
    :param m: 周期
    :return: 返回轨迹点
    '''
    allPoint = []
    for i in range(start, end):
        allPoint.extend(get_Data_From_Trajectory("SIM//EP" + str(m) + "//" + str(i) + ".txt"))
    allPoint = list(set(allPoint))
    return allPoint


def writeDataForEveryP(path, points):
    file = open(path, 'ab+')
    block_num = math.ceil(len(points) / DataBlock.BlockSize)  # 块数
    DataBlock.Block_num.append(block_num)
    block_id = 0
    while block_id < block_num:
        for i in range(block_id * DataBlock.BlockSize, (block_id + 1) * DataBlock.BlockSize):
            if i < len(points):
                p = points[i]
                b1 = st.pack('d', p.x)
                b2 = st.pack('d', p.y)
                b3 = st.pack('d', p.time)
                file.write(b1)
                file.write(b2)
                file.write(b3)
            else:
                b1 = st.pack('d', 0)
                b2 = st.pack('d', 0)
                b3 = st.pack('d', 0)
                file.write(b1)
                file.write(b2)
                file.write(b3)
        block_id += 1
    file.close()


def construct_model(region_num, error, row, col, P):
    '''
    构建模型
    :param region_num: 区域数
    :param error: 给定误差
    :param row: 行数
    :param col: 列数
    :param P: 周期
    :return:
    '''
    point_list = get_NextP_Point(0, 200, P)
    time_split = divide_region(region_num, point_list)  # 按照时间排序point_list,获取时间分割点
    t1 = time.time()
    B = B_tree.BTree(5)
    for i in range(len(time_split)):
        B.insert(B.root, time_split[i][1])
    B.printTree(B.root)  # 打印树
    B.join()
    model_num = 0  # 线性模型的个数
    for i in range(len(time_split)):
        start_b = 0
        if i == 0:
            point_order = lebe_function(row, col, point_list[:time_split[0][0] + 1])
        else:
            start_b += sum(DataBlock.Block_num)
            point_order = lebe_function(row, col, point_list[time_split[i - 1][0] + 1:time_split[i][0] + 1])
        # print("数据点个数", len(point_order))
        writeDataForEveryP(File_N.path, point_order)
        test_x = []
        test_y = []
        for j in range(len(point_order)):
            test_x.append(point_order[j].value)
            test_y.append(point_order[j].order)
        m = Linear(test_x, test_y, error)
        m.train()
        node = B.search(B.root, time_split[i][1])
        node.children.append([start_b, sum(DataBlock.Block_num) - 1, m])
        model_num += len(m.model)
    point_order = lebe_function(row, col, point_list[time_split[len(time_split) - 1][0] + 1:])
    start_b = sum(DataBlock.Block_num)
    writeDataForEveryP(File_N.path, point_order)
    # print("数据点个数", len(point_order))
    test_x = []
    test_y = []
    for i in range(len(point_order)):
        test_x.append(point_order[i].value)
        test_y.append(point_order[i].order)
    m = Linear(test_x, test_y, error)
    m.train()
    model_num += len(m.model)
    node = B.search(B.root, time_split[len(time_split) - 1][1])
    node.children.append([start_b, sum(DataBlock.Block_num) - 1, m])
    t2 = time.time()
    print("构建时间：", t2 - t1)
    print("索引大小：", (model_num * 24 + len(B.leaf_list) * 4 + B.pointer_num * 4 + (2 * region_num - 1) * 4) / 1024, "KB")
    return B, time_split, point_list


def get_Data_From_Trajectory_real(str1, Period):
    '''
    从本地获取轨迹数据，输出为轨迹点的集合
    :param str1: 轨迹的id
    :param trajID: 轨迹的坐标
    :return:
    '''
    file = open(str1)
    Point_list = []
    timeMap = {}
    timeMap[1] = 1228060800
    timeMap[2] = 1230739200
    timeMap[3] = 1233417600
    timeMap[4] = 1235836800
    timeMap[5] = 1238515200
    timeMap[6] = 1241107200
    timeMap[7] = 1243785600
    timeMap[8] = 1246377600
    timeMap[9] = 1249056000
    timeMap[10] = 1251734400
    for line in file:
        TraList = line.split(',')
        x = float(TraList[3])
        y = float(TraList[4])
        time = float(TraList[5]) - timeMap[Period]
        if time > 2678400:
            continue
        Traid = int(TraList[0])
        tempPoint = Point(x, y, time)
        tempPoint.tra_id = Traid
        if time > Point.max_time:
            Point.max_time = time
        if time < Point.min_time:
            Point.min_time = time
        if x > Point.max_x:
            Point.max_x = x
        if x < Point.min_x:
            Point.min_x = x
        if y > Point.max_y:
            Point.max_y = y
        if y < Point.min_y:
            Point.min_y = y
        Point_list.append(tempPoint)
    return Point_list


def get_point(Period):
    '''
    :param point_num: 点的数量
    :param Period: 周期
    :return:返回数据点
    '''
    allPoint = []
    file = os.listdir("GEO//P" + str(Period))
    for dir in file:
        file_name = "GEO//P" + str(Period) + "//" + dir
        allPoint.extend(get_Data_From_Trajectory_real(file_name, Period))
    allPoint = list(set(allPoint))
    return allPoint


def construct_model_real(region_num, error, row, col, P):
    '''
    构建模型
    :param region_num: 区域数
    :param error: 给定误差
    :param row: 行数
    :param col: 列数
    :param P: 周期
    :return:
    '''
    point_list = get_point(P)
    print("数据点数：", len(point_list))
    time_split = divide_region(region_num, point_list)  # 按照时间排序point_list,获取时间分割点
    t1 = time.time()
    B = B_tree.BTree(5)
    for i in range(len(time_split)):
        B.insert(B.root, time_split[i][1])
    B.printTree(B.root)  # 打印树
    B.join()
    model_num = 0  # 线性模型的个数
    for i in range(len(time_split)):
        start_b = 0
        if i == 0:
            point_order = lebe_function(row, col, point_list[:time_split[0][0] + 1])
        else:
            start_b += sum(DataBlock.Block_num)
            point_order = lebe_function(row, col, point_list[time_split[i - 1][0] + 1:time_split[i][0] + 1])
        # print("数据点个数", len(point_order))
        writeDataForEveryP(File_N.path, point_order)
        test_x = []
        test_y = []
        for j in range(len(point_order)):
            test_x.append(point_order[j].value)
            test_y.append(point_order[j].order)
        m = Linear(test_x, test_y, error)
        m.train()
        node = B.search(B.root, time_split[i][1])
        node.children.append([start_b, sum(DataBlock.Block_num) - 1, m])
        model_num += len(m.model)
    point_order = lebe_function(row, col, point_list[time_split[len(time_split) - 1][0] + 1:])
    start_b = sum(DataBlock.Block_num)
    writeDataForEveryP(File_N.path, point_order)
    # print("数据点个数", len(point_order))
    test_x = []
    test_y = []
    for i in range(len(point_order)):
        test_x.append(point_order[i].value)
        test_y.append(point_order[i].order)
    m = Linear(test_x, test_y, error)
    m.train()
    model_num += len(m.model)
    node = B.search(B.root, time_split[len(time_split) - 1][1])
    node.children.append([start_b, sum(DataBlock.Block_num) - 1, m])
    t2 = time.time()
    print("构建时间：", t2 - t1)
    print("索引大小：", (model_num * 24 + len(B.leaf_list) * 4 + B.pointer_num * 4 + (2 * region_num - 1) * 4) / 1024, "KB")
    return B, time_split, point_list


def ReadBlockByIdForInsert(id):
    '''
    增量插入时，用的读取快
    :param id:
    :return:
    '''
    tempfile = open(File_N.path, 'r+b')
    tempfile.seek(id * 8184)
    pointList = []
    tempStr = tempfile.read(8184)
    for i in range(DataBlock.BlockSize):
        mm = st.unpack('ddd', tempStr[i * 24:i * 24 + 24])
        p = Point(mm[0], mm[1], mm[2])
        pointList.append(p)
    tempfile.close()
    return pointList


def IsBlockInBuffer(BlockId):
    '''
    判断是否在缓冲区，如果小于128入队，否则队头出队，然后当前id入队
    :param BlockId: 当前访问块号
    :return:
    '''
    if BlockId in buffer.InsertId:
        return buffer.InsertTest[buffer.InsertId.index(BlockId)]
    else:
        if len(buffer.InsertTest) < 128:
            buffer.InsertBlockNum += 1
            temp = ReadBlockByIdForInsert(BlockId)
            buffer.InsertTest.append(temp)
            buffer.InsertId.append(BlockId)
            return temp
        else:
            buffer.InsertBlockNum += 1
            PointList = buffer.InsertTest.pop(0)
            PointListId = buffer.InsertId.pop(0)
            temp = ReadBlockByIdForInsert(BlockId)
            buffer.InsertTest.append(temp)
            buffer.InsertId.append(BlockId)
            return temp


def train_first(tree, P, time_split, row, col):
    '''
    "一期训练，后期不训练"
    :param tree: 分段线性回归树
    :param P: 周期
    :param time_split: 时间分割点
    :param row: 网格行数
    :param col: 网格列数
    :return:块模型
    '''
    File_N.path = 'EP2.dat'
    DataBlock.Block_num.clear()
    point_list = get_NextP_Point(0, 300, P)  # 读取模拟数据
    point_region = defaultdict(list)  # 按区域存储数据，键为区域号
    splits = []  # 时间分割值列表
    for l in time_split:
        splits.append(l[1])
    for p in point_list:
        region_id = bisect_left(splits, p.time)  # 数据点的区域号
        p.region_id = region_id
        point_region[region_id].append(p)
    # 使用分段线性模型得到数据点的预测值并存储
    id = 0  # 区域号
    block_split = {}  # 键为区域号，值为块的分割值
    children_num = []  # 每个叶子结点拥有的孩子个数，用于计算区域号
    avg_err = []  # 平均误差
    for leaf in tree.leaf_list:
        children_num.append(len(leaf.children))
        for i in range(len(leaf.children)):
            temp = leaf.children[i]
            model = temp[2]
            point_order = lebe_function(row, col, point_region[id])
            X = []
            for p in point_order:
                X.append(p.value)
            pre = model.predict(X)  # 该区域中数据点的预测值
            sum_err = 0
            for j in range(len(point_order)):
                point_order[j].predict_value = pre[j]
                sum_err += abs(point_order[j].order - pre[j])
            # print("平均误差：", sum_err / len(X))
            avg_err.append(sum_err / len(X))
            point_order.sort(key=lambda p: p.predict_value)
            writeDataForEveryP(File_N.path, point_order)
            temp_block = []
            count = 0
            while count < DataBlock.Block_num[-1]:
                temp_block.append(point_order[count * DataBlock.BlockSize].predict_value)
                count += 1
            block_split[id] = temp_block
            id += 1
    print("数据块数：", sum(DataBlock.Block_num), "区域个数：", len(block_split))
    return block_split, point_list, children_num, avg_err


def conflictTest(tree, P, time_split, row, col, eb):
    point_list = get_NextP_Point(0, 200, P)  # 读取模拟数据
    splits = []  # 时间分割值列表
    for l in time_split:
        splits.append(l[1])
    point_region = defaultdict(list)  # 按区域存储数据，键为区域号
    for p in point_list:
        region_id = bisect_left(splits, p.time)  # 数据点的区域号
        p.region_id = region_id
        point_region[region_id].append(p)
    id = 0  # 区域号
    conNum = 0  # 冲突次数
    K = 5
    for leaf in tree.leaf_list:
        for i in range(len(leaf.children)):
            length = math.ceil(len(point_region[id]) / DataBlock.BlockSize) * DataBlock.BlockSize
            arr = [0 for k in range(length)]
            temp = leaf.children[i]
            model = temp[2]
            point_order = lebe_function(row, col, point_region[id])
            X = []
            for p in point_order:
                X.append(p.value)
            pre = model.predict(X)  # 该区域中数据点的预测值
            factor = len(point_region[id]) / DataBlock.Block_num[id]
            for j in range(len(pre)):
                addr = int(pre[j] * factor)
                if addr < 0: addr = 0
                if addr >= len(arr): addr = len(arr) - 1
                if arr[addr] == 1:
                    start = addr - eb
                    if start < 0: start = 0
                    end = addr + eb
                    if end >= len(arr): end = len(arr) - 1
                    pos = random.randint(start, end)  # 在误差范围内随机选择位置
                    c = 0
                    while arr[pos] == 1 and c < K:
                        pos = random.randint(start, end)
                        c += 1
                    if c == K:
                        conNum += 1
                    else:
                        arr[pos] = 1
                else:
                    arr[addr] = 1
            id += 1
    print("冲突率：", conNum / len(point_list))


def train_first_real(tree, P, time_split, row, col):
    '''
    "一期训练，后期不训练"
    :param tree: 分段线性回归树
    :param P: 周期
    :param time_split: 时间分割点
    :param row: 网格行数
    :param col: 网格列数
    :return:块模型
    '''
    File_N.path = 'EP2.dat'
    DataBlock.Block_num.clear()
    point_list = get_point(P)
    print("数据点数：", len(point_list))
    point_region = defaultdict(list)  # 按区域存储数据，键为区域号
    splits = []  # 时间分割值列表
    for l in time_split:
        splits.append(l[1])
    for p in point_list:
        region_id = bisect_left(splits, p.time)  # 数据点的区域号
        p.region_id = region_id
        point_region[region_id].append(p)
    # 使用分段线性模型得到数据点的预测值并存储
    id = 0  # 区域号
    block_split = {}  # 键为区域号，值为块的分割值
    children_num = []  # 每个叶子结点拥有的孩子个数，用于计算区域号
    avg_err = []  # 平均误差
    for leaf in tree.leaf_list:
        children_num.append(len(leaf.children))
        for i in range(len(leaf.children)):
            if id not in point_region.keys():
                id += 1
                continue
            temp = leaf.children[i]
            model = temp[2]
            point_order = lebe_function(row, col, point_region[id])
            X = []
            for p in point_order:
                X.append(p.value)
            pre = model.predict(X)  # 该区域中数据点的预测值
            sum_err = 0
            for j in range(len(point_order)):
                point_order[j].predict_value = pre[j]
                sum_err += abs(point_order[j].order - pre[j])
            # print("平均误差：", sum_err / len(X))
            avg_err.append(sum_err / len(X))
            point_order.sort(key=lambda p: p.predict_value)
            writeDataForEveryP(File_N.path, point_order)
            temp_block = []
            count = 0
            while count < DataBlock.Block_num[-1]:
                temp_block.append(point_order[count * DataBlock.BlockSize].predict_value)
                count += 1
            block_split[id] = temp_block
            id += 1
    print("数据块数：", sum(DataBlock.Block_num))
    return block_split, point_list, children_num, avg_err


def point_query(tree, q):
    '''
    点查询
    :param tree:
    :param q: 轨迹点
    :return:
    '''
    node = tree.search(tree.root, q.time)
    ind = bisect_left(node.keys, q.time)
    temp = node.children[ind]
    model = temp[2]
    pre = model.predict([q.value])
    s = temp[0] + int(pre[0] - model.error) // DataBlock.BlockSize
    e = temp[0] + int(pre[0] + model.error) // DataBlock.BlockSize
    if s < temp[0]:
        s = temp[0]
    if e > temp[1]:
        e = temp[1]
    return [s, e]


def point_query1(tree, q, block_split):
    '''
    点查询
    :param tree:
    :param q: 轨迹点
    :return:
    '''
    node = tree.search(tree.root, q.time)
    ind = bisect_left(node.keys, q.time)
    temp = node.children[ind]
    model = temp[2]
    pre = model.predict([q.value])
    temp_block = block_split[q.region_id]
    s = sum(DataBlock.Block_num[:q.region_id]) + bisect_left(temp_block, pre[0]) - 1  # 起始块号
    return [s, s + 1]


def value_segments(w, row, col):
    '''
    求查询窗口的映射值连续段
    :param w: [x1,x2,y1,y2]
    :param row: 网格行数
    :param col: 网格列数
    :return:
    '''
    inter_x = (Point.max_x - Point.min_x) / col
    inter_y = (Point.max_y - Point.min_y) / row
    area = inter_y * inter_x
    begin = []
    end = []
    # 左下角(r,c)
    r = (w[2] - Point.min_y) // inter_y  # 行
    if r >= row:
        r = row - 1
    c = (w[0] - Point.min_x) // inter_x  # 列
    if c >= col:
        c = col - 1
    cell = r * col + c  # 网格序号
    cell_x = Point.min_x + inter_x * c
    cell_y = Point.min_y + inter_y * r
    value = cell + (w[2] - cell_y) * (w[0] - cell_x) / area
    begin.append(value)
    # 右上角(r1,c1)
    r1 = (w[3] - Point.min_y) // inter_y  # 行
    if r1 >= row:
        r1 = row - 1
    c1 = (w[1] - Point.min_x) // inter_x  # 列
    if c1 >= col:
        c1 = col - 1
    cell1 = r1 * col + c1  # 网格序号
    cell_x1 = Point.min_x + inter_x * c1
    cell_y1 = Point.min_y + inter_y * r1
    value1 = cell1 + (w[3] - cell_y1) * (w[1] - cell_x1) / area
    # 起始值
    for i in range(int(r + 1), int(r1 + 1)):
        cell2 = i * col + c
        begin.append(cell2)
    # 终止值
    for i in range(int(r), int(r1)):
        cell2 = i * col + c1
        cell_x2 = Point.min_x + inter_x * c1
        value2 = cell2 + (w[1] - cell_x2) / inter_x
        end.append(value2)
    end.append(value1)
    return begin, end


def range_search(tree, q, row, col):
    '''
    范围查询
    :param tree:
    :param q: [x1,x2,y1,y2,t1,t2]
    :return:
    '''
    model_list = []
    start_node = tree.search(tree.root, q[4])
    end_node = tree.search(tree.root, q[5])
    start_ind = bisect_left(start_node.keys, q[4])
    i = start_ind
    while start_node != end_node:
        while i < len(start_node.children):
            temp = start_node.children[i]
            model_list.append(temp)
            i += 1
        start_node = start_node.pointer
        i = 0
    end_ind = bisect_left(end_node.keys, q[5])
    while i <= end_ind:
        temp = end_node.children[i]
        model_list.append(temp)
        i += 1
    begin, end = value_segments(q[:4], row, col)
    block_id = set()
    for vl in model_list:
        for j in range(len(begin)):
            pre1 = vl[2].predict([begin[j]])
            pre2 = vl[2].predict([end[j]])
            s = vl[0] + int(pre1[0] - vl[2].error) // DataBlock.BlockSize
            e = vl[0] + int(pre2[0] + vl[2].error) // DataBlock.BlockSize
            if s < vl[0]:
                s = vl[0]
            if e > vl[1]:
                e = vl[1]
            for k in range(s, e + 1):
                block_id.add(k)
    return block_id


def range_search1(tree, q, row, col, block_split, children_num):
    '''
    范围查询1
    :param tree:分段线性回归树
    :param q: [x1,x2,y1,y2,t1,t2]
    :param block_split:块分割值
    :param children_num:每个叶子结点拥有的孩子数
    :return:
    '''
    model_list = []
    start_node = tree.search(tree.root, q[4])
    ind = tree.leaf_list.index(start_node)  # 第几个孩子结点
    end_node = tree.search(tree.root, q[5])
    start_ind = bisect_left(start_node.keys, q[4])
    start_region = sum(children_num[:ind]) + start_ind  # 起始区域号
    # 获取与时间范围相交的区域的模型
    i = start_ind
    while start_node != end_node:
        while i < len(start_node.children):
            temp = start_node.children[i]
            model_list.append(temp)
            i += 1
        start_node = start_node.pointer
        i = 0
    end_ind = bisect_left(end_node.keys, q[5])
    while i <= end_ind:
        temp = end_node.children[i]
        model_list.append(temp)
        i += 1
    begin, end = value_segments(q[:4], row, col)  # 获取映射值连续段
    block_id = set()
    for vl in model_list:
        if start_region not in block_split.keys():
            start_region += 1
            continue
        temp_block = block_split[start_region]
        for j in range(len(begin)):
            pre1 = vl[2].predict([begin[j]])
            pre2 = vl[2].predict([end[j]])
            s = sum(DataBlock.Block_num[:start_region]) + bisect_left(temp_block, pre1[0]) - 1  # 起始块号
            if s < 0:
                s = 0
            e = sum(DataBlock.Block_num[:start_region]) + bisect_left(temp_block, pre2[0]) + 1  # 终止块号
            if e >= sum(DataBlock.Block_num):
                e = sum(DataBlock.Block_num) - 1
            for k in range(s, e + 1):
                block_id.add(k)
        start_region += 1
    return block_id


def read_range(file_name):
    range_query = []
    tempData = []
    fo = open(file_name)
    line = fo.readline()
    while line:
        tempData.append(line)
        line = fo.readline()
    fo.close()
    splitData = []
    for i in range(len(tempData)):
        splitData.append(tempData[i].split(","))
    for j in range(len(splitData)):
        x_min = float(splitData[j][0])
        x_max = float(splitData[j][1])
        y_min = float(splitData[j][2])
        y_max = float(splitData[j][3])
        time_min = float(splitData[j][4])
        time_max = float(splitData[j][5])
        range_query.append([x_min, x_max, y_min, y_max, time_min, time_max])
    return range_query


def point_filter(points, block_split, tree):
    '''
    查找相应块
    :param points: 数据点
    :return:
    '''
    recall = 0
    t1 = time.time()
    for i in range(400):
        num = random.randint(0, len(points))
        temp = point_query1(tree, points[num], block_split)
        for id in range(temp[0], temp[1] + 1):
            point_list = IsBlockInBuffer(id)
            flag = False
            for p in point_list:
                if p.x == points[num].x and p.y == points[num].y and p.time == points[num].time:
                    recall += 1
                    flag = True
                    break
            if flag:
                break
    t2 = time.time()
    print("召回率", recall / 400, "点查询平均访问块数:", buffer.InsertBlockNum / 400)
    print("查询时间：", t2 - t1)


def point_filter2(points, tree):
    '''
    查找相应块
    :param points: 数据点
    :return:
    '''
    recall = 0
    q = []  # 查询的点
    for i in range(400):
        num = random.randint(0, len(points))
        while points[num] in q:
            num = random.randint(0, len(points))
        q.append(points[num])
    t1 = time.time()
    for i in range(len(q)):
        temp = point_query(tree, q[i])
        for id in range(temp[0], temp[1] + 1):
            point_list = IsBlockInBuffer(id)
            flag = False
            for p in point_list:
                if p.x == q[i].x and p.y == q[i].y and p.time == q[i].time:
                    recall += 1
                    flag = True
                    break
            if flag:
                break
    t2 = time.time()
    print("召回率", recall / 400, "点查询平均访问块数:", buffer.InsertBlockNum / 400)
    print("查询时间：", t2 - t1)


def sim_test(row, col, error, region_num):
    '''
    在sim上测试模型
    :param row: 网格行数
    :param col: 网格列数
    :param error: 给定误差
    :param region_num: 区域个数
    :return:
    '''
    B, time_split, points = construct_model(region_num, error, row, col, 1)
    conflictTest(B, 2, time_split, row, col, error)  # 测试冲突
    block_split, point_list, children_num, avg_err = train_first(B, 2, time_split, row, col)
    # --------------点查询1------------------
    print("开始点查询:")
    point_filter(point_list, block_split, B)
    # --------------点查询2------------------
    # point_filter2(points, B)
    # 范围查询
    # recall=0
    # real_num=0
    # range_list=read_range('SIM//regionForSim.txt')
    # t_sum=0
    # for q in range_list[:500]:
    #     for p in point_list:
    #         if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
    #             real_num += 1
    #     t1=time.time()
    #     block_id=range_search1(B,q,row,col,block_split,children_num)
    #     for id in block_id:
    #         point_list1 = IsBlockInBuffer(id)
    #         for p in point_list1:
    #             if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
    #                 recall += 1
    #     t2 = time.time()
    #     t_sum += (t2 - t1)
    # print("召回率：",recall/real_num,"平均磁盘访问块数",buffer.InsertBlockNum/500)
    # print("平均查询时间：",t_sum/500)
    # --------------范围查询2-------------------------
    recall = 0
    real_num = 0
    range_list = read_range('SIM//regionForSim.txt')
    t_sum = 0
    for q in range_list[:500]:
        for p in points:
            if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
                real_num += 1
        t1 = time.time()
        block_id = range_search(B, q, row, col)
        for id in block_id:
            point_list = IsBlockInBuffer(id)
            for p in point_list:
                if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
                    recall += 1
        t2 = time.time()
        t_sum += (t2 - t1)
    print("召回率：", recall / real_num, "平均磁盘访问块数", buffer.InsertBlockNum / 500)
    print("平均查询时间：", t_sum / 500)


def geo_test(row, col, error, region_num):
    '''
    在geo上测试模型
    :param row: 网格行数
    :param col: 网格列数
    :param error: 给定误差
    :param region_num: 区域个数
    :return:
    '''
    B, time_split, points = construct_model_real(region_num, error, row, col, 1)
    block_split, point_list, children_num, avg_err = train_first_real(B, 2, time_split, row, col)
    # --------------点查询1------------------
    print("开始点查询:")
    point_filter(point_list, block_split, B)
    # --------------点查询2------------------
    # point_filter2(points, B)
    # 范围查询
    # recall=0
    # real_num=0
    # range_list=read_range('GEO//regionForRealFor5wOne.txt')
    # t_sum=0
    # for q in range_list[:400]:
    #     for p in point_list:
    #         if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
    #             real_num += 1
    #     t1=time.time()
    #     block_id=range_search1(B,q,row,col,block_split,children_num)
    #     for id in block_id:
    #         point_list1 = IsBlockInBuffer(id)
    #         for p in point_list1:
    #             if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
    #                 recall += 1
    #     t2 = time.time()
    #     t_sum += (t2 - t1)
    # print("召回率：",recall/real_num,"平均磁盘访问块数",buffer.InsertBlockNum/400)
    # --------------范围查询2-------------------------
    recall = 0
    real_num = 0
    range_list = read_range('GEO//regionForRealFor5wOne.txt')
    t_sum = 0
    for q in range_list[:500]:
        for p in points:
            if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
                real_num += 1
        t1 = time.time()
        block_id = range_search(B, q, row, col)
        for id in block_id:
            point_list = IsBlockInBuffer(id)
            for p in point_list:
                if p.x >= q[0] and p.x <= q[1] and p.y >= q[2] and p.y <= q[3] and p.time >= q[4] and p.time <= q[5]:
                    recall += 1
        t2 = time.time()
        t_sum += (t2 - t1)
    print("召回率：", recall / real_num, "平均磁盘访问块数", buffer.InsertBlockNum / 500)
    print("平均查询时间：", t_sum / 500)


if __name__ == "__main__":
    sim_test(40, 40, 1200, 25)  # 在模拟数据集上测试分段线性回归树
    geo_test(40, 40, 1200, 25)  # 在真实数据集上测试分段线性回归树
