import time
from operator import attrgetter
import numpy as np
import struct as st
import math
import os

# ep
ep = 1
DATA = "GEO"
max_c = 512  # 最大桶容量
flag = 0  # 表示分裂次数，用来循环分裂操作flag % 2 == 0时分裂x；flag % 2 == 1时分裂y
# range_query_file_name = "../SIM_DATA/range_sim1.txt"
nx = 178  #
ny = nx  # grid array 大小
print("nx=" + str(nx))
max_x_value = 64000  # x
min_x_value = 0
max_y_value = 2678400  # y
min_y_value = 0


# 建立3维栅格数组
# 3维栅格数组，数组值为桶号，初始为桶0
grid_array = np.arange(nx * ny, dtype='int64').reshape(nx, ny)
# 建立线性刻度，方便查找
X = [min_x_value]  # 线性刻度
Y = [min_y_value]
# 建立B数组，用来装桶
B = []
num = 0
visit_time = 0
bucket_count = 0
write_time = 0
data_file_name = DATA + "/models/EP" + str(ep) + "/sim.dat"
index_file_name = DATA + "/models/EP" + str(ep) + "/sim.idx"


class Record:
    x = 0.0
    y = 0.0


class Position:
    x = 0
    y = 0


class Range:
    x_min = 0.0
    x_max = 0.0
    y_min = 0.0
    y_max = 0.0


# 块类
class Block:
    id = 0
    list = []


# 缓存区类
class Buffer:
    id_list = []  # 用于存放块号，用于先进先出
    block_dict = {}  # 块号：块
    max_size = 0

    def append(self, block):
        """
        将块添加入缓存区
        :param block: 块；类
        :return: 无
        """
        global write_time
        if len(self.id_list) < self.max_size:
            self.id_list.append(block.id)
            self.block_dict[block.id] = block
        else:
            out_block_id = self.id_list[0]
            del self.id_list[0]
            write_time += 1
            del self.block_dict[out_block_id]
            self.id_list.append(block.id)
            self.block_dict[block.id] = block


def write_block(block, data_file):
    """
    写数据块
    :param block: 块
    :return: 无
    """
    global max_c
    data_file.seek(block.id * 8 * 1024)
    for record in block.list:
        b_x = st.pack("d", record.x)
        b_y = st.pack("d", record.y)
        data_file.write(b_x)
        data_file.write(b_y)
    for i in range(max_c - len(block.list)):
        data_file.write(st.pack("d", 0))
        data_file.write(st.pack("d", 0))


def find(r):
    """
    在线性刻度上查找记录
    :param r: 点
    :return: grid_array中的三维坐标
    """
    pos = Position()
    for i in range(nx):
        if r.x < X[i]:
            pos.x = i - 1
            break
    for i in range(ny):
        if r.y < Y[i]:
            pos.y = i - 1
            break
    return pos


def insert(r):
    """
    将记录r插入栅格文件中
    :param r: 点
    :return: 无
    """
    global flag     # 声明使用全局变量flag
    pos = find(r)
    bucket_id = grid_array[pos.x][pos.y]
    if len(B[bucket_id].list) <= max_c:
        B[bucket_id].list.append(r)
    elif len(B[bucket_id].list) > max_c:
        flag += 1


def point_query(r):
    """
    点查询
    :param r: 点
    :return: 找到1，没找到0
    """
    global visit_time
    global index_buffer
    global data_buffer
    global data_file_name
    global nx, ny
    # 根据linear scale找到grid array中的位置
    pos = find(r)
    # 根据grid array中的位置，计算出索引块号以及块中位置
    block_id = int((pos.x * ny + pos.y) / 1024)
    pos_in_block = pos.x * ny + pos.y - block_id * 1024
    # 现在缓冲区中查找索引块
    find_block = 0
    if len(index_buffer.id_list) != 0 and block_id in index_buffer.block_dict:
        block = index_buffer.block_dict[block_id]
        bucket_id = block.list[pos_in_block]
        find_block = 1
    # 如果缓冲区中没找到，则去硬盘读取索引块
    if find_block == 0:
        with open(index_file_name, 'rb') as index_file:
            visit_time += 1
            index_file.seek(block_id * 8 * 1024)
            index_str = index_file.read(8 * 1024)
            index_block = Block()
            index_block.id = block_id
            index_block.list = []
            for i in range(1024):
                if len(index_str[i * 8:i * 8 + 8]) != 8:
                    break
                index_tuple = st.unpack("q", index_str[i*8:i*8+8])
                index_block.list.append(index_tuple[0])
            bucket_id = index_block.list[pos_in_block]
            index_buffer.append(index_block)
    # 在缓存中寻找数据块，如果找到直接在缓存中的数据块中查找点
    if len(data_buffer.id_list) != 0 and bucket_id in data_buffer.block_dict:
        bucket = data_buffer.block_dict[bucket_id]
        for record in bucket.list:
            if record.x == r.x and record.y == r.y:
                return 1
        return 0
    # 在缓存中没找到数据块，则到硬盘上找，并加入缓存中，然后查找点
    with open(data_file_name, 'rb') as data_file:
        visit_time += 1
        # data_file.seek(0)
        # for i in range(bucket_id - 1):
        #     data_file.seek(max_c * 24 + 8, 1)  # 每个桶存max_c*24B的数据，即max_c个点
        data_file.seek(bucket_id * 8 * 1024)
        str = data_file.read(8 * 1024)
        bucket = Block()
        bucket.id = bucket_id
        bucket.list = []
        for i in range(max_c):
            if len(str[i * 16:i * 16 + 16]) != 16:
                break
            record_str = st.unpack('dd', str[i*16:i*16+16])
            if record_str[0] == 0 and record_str[1] == 0:
                break
            record = Record()
            record.x = record_str[0]
            record.y = record_str[1]
            bucket.list.append(record)
        data_buffer.append(bucket)
        for record in bucket.list:
            if record.x == r.x and record.y == r.y:
                return 1
    return 0


def range_query(rang, data_buffer, index_buffer):
    """
    范围查询
    :param rang: 范围
    :return: 找到的点数组
    """
    global visit_time
    global bucket_count
    bucket_id_set = set()
    pos_list = []
    range_array = []
    # grid array中结束点的位置
    r = Record()
    r.x = rang.x_max
    r.y = rang.y_max
    end_pos = find(r)
    # grid array中开始点的位置
    r = Record()
    r.x = rang.x_min
    r.y = rang.y_min
    start_pos = find(r)
    # 将范围中每个点的位置存入位置数组
    for i in range(start_pos.x, end_pos.x + 1):
        for j in range(start_pos.y, end_pos.y + 1):
                pos = Position()
                pos.x = i
                pos.y = j
                pos_list.append(pos)
    # 遍历位置数组，在索引块中查找并加入桶号集合
    for pos in pos_list:
        # 根据grid array中的位置，计算出索引块号以及块中位置
        block_id = int((pos.x * ny + pos.y) / 1024)
        pos_in_block = pos.x * ny + pos.y - block_id * 1024
        # 现在缓冲区中查找索引块
        find_block = 0
        if len(index_buffer.id_list) != 0 and block_id in index_buffer.block_dict:
            block = index_buffer.block_dict[block_id]
            bucket_id = block.list[pos_in_block]
            find_block = 1
        # 如果缓冲区中没找到，则去硬盘读取缓存块
        if find_block == 0:
            with open(index_file_name, 'rb') as index_file:
                visit_time += 1
                index_file.seek(block_id * 8 * 1024)
                index_str = index_file.read(8 * 1024)
                index_block = Block()
                index_block.id = block_id
                index_block.list = []
                for i in range(1024):
                    if len(index_str[i * 8:i * 8 + 8]) != 8:
                        break
                    id = st.unpack("q", index_str[i*8:i*8+8])
                    index_block.list.append(id[0])
                bucket_id = index_block.list[pos_in_block]
                bucket_id_set.add(bucket_id)
                index_buffer.append(index_block)
        bucket_id_set.add(bucket_id)
    bucket_count += len(bucket_id_set)
    # 在桶号集合中查找，遍历每一个桶，查找范围中的点
    for bucket_id in bucket_id_set:
        # 在缓存中寻找数据块，如果找到直接在缓存中的数据块中查找点
        find_block = 0
        if len(data_buffer.id_list) != 0 and bucket_id in data_buffer.block_dict:
            find_block = 1
            bucket = data_buffer.block_dict[bucket_id]
            for record in bucket.list:
                if (rang.x_min <= record.x <= rang.x_max) and (rang.y_min <= record.y <= rang.y_max):
                    range_array.append(record)
        # 在缓存中没找到数据块，则到硬盘上找，并加入缓存中，然后查找点
        if find_block == 0:
            # read_time_1 = time.time()
            with open(data_file_name, 'rb') as data_file:
                visit_time += 1
                # data_file.seek(0)
                # for i in range(bucket_id - 1):
                #     data_file.seek(max_c * 24 + 8, 1)  # 每个桶存max_c*24B的数据，即max_c个点
                data_file.seek(bucket_id * 8 * 1024)
                str = data_file.read(8 * 1024)
                # read_time_2 = time.time()
                # read_time = read_time_2 - read_time_1
                # print(read_time, "s")
                bucket = Block()
                bucket.id = bucket_id
                bucket.list = []
                data_buffer.append(bucket)
                for i in range(max_c):
                    record_str = st.unpack('dd', str[i * 16:i * 16 + 16])
                    if record_str[0] == 0 and record_str[1] == 0:
                        break
                    record = Record()
                    record.x = record_str[0]
                    record.y = record_str[1]
                    bucket.list.append(record)
                    if (rang.x_min <= record.x <= rang.x_max) and (rang.y_min <= record.y <= rang.y_max):
                        range_array.append(record)
    return range_array


def Grid_File_Geo():
    global num, nx, ny
    global visit_time
    # 建立模型
    for i in range(nx):
        for j in range(ny):
            bucket = Block()
            bucket.id = num
            num += 1
            bucket.list = []
            B.append(bucket)

    # 第一期静态装入
    data_buffer = Buffer()
    data_buffer.max_size = 128
    data_buffer.id_list = []
    data_buffer.block_dict = {}
    index_buffer = Buffer()
    index_buffer.max_size = 128
    index_buffer.id_list = []
    index_buffer.block_dict = {}
    visit_time = 0
    write_time = 0
    count = 0
    R = []
    # print('ep' + str(ep))
    time_start = time.time()
    print(DATA)
    file_path = "../" + DATA + "/EP" + str(ep) + "/"
    dirs = os.listdir(file_path)
    for file_name in dirs:
        with open(file_path + "/" + file_name) as file_object:
            for line in file_object:
                line = line.rstrip()
                line = line.split(",")
                r = Record()
                r.x = float(line[3])
                r.y = float(line[4])
                R.append(r)
    R = list(set(R))
    max_count = len(R)
    print(max_count)
    visit_time += math.ceil(max_count / max_c)
    R.sort(key=attrgetter("x"))
    for i in range(1, nx):
        X.append(R[i * int(max_count / nx)].x)
    X.append(max_x_value)
    print(X)
    R.sort(key=attrgetter("y"))
    for i in range(1, ny):
        Y.append(R[i * int(max_count / ny)].y)
    Y.append(max_y_value)
    print(Y)
    for r in R:
        insert(r)
    print('flag = ' + str(flag))
    # 将grid array 写入硬盘
    index_file = open(index_file_name, 'wb')
    for i in range(nx):
        for j in range(ny):
            bucket_id = st.pack("q", grid_array[i][j])
            index_file.write(bucket_id)
    visit_time += math.ceil((nx * ny) / 1024)
    index_file.close()
    # 将桶中数据写入硬盘
    data_file = open(data_file_name, 'wb')
    write_time_start = time.time()
    for i in range(num):
        write_block(B[i], data_file)
        visit_time += 1
    write_time_end = time.time()
    data_file.close()
    time_end = time.time()
    print('cost of build time:', time_end - time_start, 's')
    print('visit time:', visit_time)
    print("\n")

    if ep <= 2:
        # 一个索引块可以放1024(1000)个索引项 1000 * 8
        # 一个数据块可以放341个数据
        # 范围查询
        print("range query")
        data_buffer = Buffer()
        data_buffer.max_size = 128
        data_buffer.id_list = []
        data_buffer.block_dict = {}
        index_buffer = Buffer()
        index_buffer.max_size = 640
        index_buffer.id_list = []
        index_buffer.block_dict = {}
        record_sum = 0
        visit_time = 0
        time_start = time.time()
        # 将索引块加入缓存区
        index_file = open(index_file_name, "rb")
        data_file = open(data_file_name, "rb")
        ranges = []
        with open("../" + DATA + "/range_geo" + ".txt") as file_object:
            for line in file_object:
                line = line.rstrip()
                line = line.split(",")
                rang = Range()
                rang.x_min = float(line[0])
                rang.x_max = float(line[1])
                rang.y_min = float(line[2])
                rang.y_max = float(line[3])
                ranges.append(rang)
        if ep == 1:
            for rang in ranges[:250]:
                result = range_query(rang, data_buffer, index_buffer)
                record_count = len(result)
                record_sum += record_count
        if ep == 2:
            for rang in ranges[250:]:
                result = range_query(rang, data_buffer, index_buffer)
                record_count = len(result)
                record_sum += record_count
        time_end = time.time()
        print('range_query_time cost:', time_end - time_start, 's')
        print('visit time:', visit_time)
        print('record sum:', record_sum)
        print('bucket count:', bucket_count)
        print("\n")

    # # 一个索引块可以放1024(1000)个索引项 1000 * 8
    # # 一个数据块可以放341个数据
    # # 范围查询
    # print("range query")
    # data_buffer = Buffer()
    # data_buffer.max_size = 128
    # data_buffer.id_list = []
    # data_buffer.block_dict = {}
    # index_buffer = Buffer()
    # index_buffer.max_size = 640
    # index_buffer.id_list = []
    # index_buffer.block_dict = {}
    # record_sum = 0
    # visit_time = 0
    # write_time = 0
    # bucket_count = 0
    # time_start = time.time()
    # # 将索引块加入缓存区
    # index_file = open(index_file_name, "rb")
    # data_file = open(data_file_name, "rb")
    # ranges1 = []
    # ranges = []
    # with open("../" + DATA + "/range_geo" + ".txt") as file_object:
    #     for line in file_object:
    #         line = line.rstrip()
    #         line = line.split(",")
    #         rang = Range()
    #         rang.x_min = float(line[0])
    #         rang.x_max = float(line[1])
    #         rang.y_min = float(line[2])
    #         rang.y_max = float(line[3])
    #         ranges1.append(rang)
    # ranges.extend(ranges1[:5])
    # ranges.extend(ranges1[125:130])
    # ranges.extend(ranges1[250:255])
    # ranges.extend(ranges1[375:380])
    # print(len(ranges))
    # for rang in ranges:
    #     result = range_query(rang)
    #     record_count = len(result)
    #     record_sum += record_count
    # time_end = time.time()
    # print('range_query_time cost:', time_end - time_start, 's')
    # print('visit time:', visit_time)
    # print('record sum:', record_sum)
    # print('bucket count:', bucket_count)
    # print("\n")


if __name__ == "__main__":
    Grid_File_Geo()
