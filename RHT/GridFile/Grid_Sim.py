import time
from operator import attrgetter
import numpy as np
import struct as st
import math


class Record:
    x = 0.0
    y = 0.0
    z = 0.0


class Position:
    x = 0
    y = 0
    z = 0


class Range:
    x_min = 0.0
    x_max = 0.0
    y_min = 0.0
    y_max = 0.0
    z_min = 0.0
    z_max = 0.0


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


def write_block(block):
    """
    写数据块
    :param block: 块
    :return: 无
    """
    global data_file
    global max_c
    data_file.seek(block.id *(max_c * 24 + 8))
    for record in block.list:
        b_x = st.pack("d", record.x)
        b_y = st.pack("d", record.y)
        b_z = st.pack("d", record.z)
        data_file.write(b_x)
        data_file.write(b_y)
        data_file.write(b_z)
    for i in range(max_c - len(block.list)):
        data_file.write(st.pack("d", 0))
        data_file.write(st.pack("d", 0))
        data_file.write(st.pack("d", 0))
    data_file.write(st.pack("d", 0))


def split_x(pos):
    """
    按x坐标分裂
    :param pos: grid array上的位置
    :return: 无
    """
    global X
    global Y
    global Z
    global B
    global index_file_name
    global num
    global nx
    global grid_array
    global visit_time
    global index_buffer
    global write_time
    old_bucket_id = grid_array[pos.x][pos.y][pos.z]
    mid_x = (X[pos.x] + X[pos.x + 1]) / 2
    X.insert(pos.x + 1, mid_x)
    nx += 1
    # 给新分割的栅格块分配新桶
    # grid array切割
    grid_array = np.insert(grid_array, pos.x, grid_array[pos.x], axis=0)
    grid_array[pos.x + 1][pos.y][pos.z] = num
    for i in range(math.ceil((nx * ny * nz) / 1024)):
        # 现在缓冲区中查找索引块
        find_block = 0
        if len(index_buffer.id_list) != 0 and i in index_buffer.block_dict:
            block = index_buffer.block_dict[i]
            # bucket_id = block.list[pos_in_block]
            find_block = 1
        # 如果缓冲区中没找到，则去硬盘读取索引块
        if find_block == 0:
            with open(index_file_name, 'rb') as index_file:
                index_file.seek(i * 8 * 1024)
                index_str = index_file.read(8 * 1024)
                index_block = Block()
                index_block.id = i
                index_block.list = []
                for i in range(1024):
                    if len(index_str[i * 8:i * 8 + 8]) != 8:
                        break
                    index_tuple = st.unpack("q", index_str[i * 8:i * 8 + 8])
                    index_block.list.append(index_tuple[0])
                # bucket_id = index_block.list[pos_in_block]
                index_buffer.append(index_block)
    visit_time += math.ceil((nx * ny * nz) / 1024)
    index_buffer.block_dict.clear()
    index_buffer.id_list.clear()
    # 将bucket中大于x_mid对应的线性刻度的记录移到bucket_new中
    y_min = Y[pos.y]
    y_max = Y[pos.y + 1]
    z_min = Z[pos.z]
    z_max = Z[pos.z + 1]
    # 添加新桶
    new_bucket = Block()
    new_bucket.id = num
    new_bucket.list = []
    num += 1
    B.append(new_bucket)
    remove_record = []
    for record in B[old_bucket_id].list:
        if X[pos.x] <= record.x < mid_x and y_min <= record.y < y_max and z_min <= record.z < z_max:  # 如果小于，则写入新桶，并删除
            B[new_bucket.id].list.append(record)
            remove_record.append(record)
    for record in remove_record:
        B[old_bucket_id].list.remove(record)
    # write_block(new_bucket)
    write_time += 1


def split_y(pos):
    """
    按y坐标分裂
    :param pos: grid array上的位置
    :return: 无
    """
    global X
    global Y
    global Z
    global B
    global index_file_name
    global num
    global ny
    global grid_array
    global visit_time
    global index_buffer
    global write_time
    old_bucket_id = grid_array[pos.x][pos.y][pos.z]
    mid_y = (Y[pos.y] + Y[pos.y + 1]) / 2
    Y.insert(pos.y + 1, mid_y)
    ny += 1
    # 给新分割的栅格块分配新桶
    # grid array切割
    grid_array = np.insert(grid_array, pos.y, grid_array[:, pos.y], axis=1)
    grid_array[pos.x][pos.y + 1][pos.z] = num
    for i in range(math.ceil((nx * ny * nz) / 1024)):
        # 现在缓冲区中查找索引块
        find_block = 0
        if len(index_buffer.id_list) != 0 and i in index_buffer.block_dict:
            block = index_buffer.block_dict[i]
            # bucket_id = block.list[pos_in_block]
            find_block = 1
        # 如果缓冲区中没找到，则去硬盘读取索引块
        if find_block == 0:
            with open(index_file_name, 'rb') as index_file:
                index_file.seek(i * 8 * 1024)
                index_str = index_file.read(8 * 1024)
                index_block = Block()
                index_block.id = i
                index_block.list = []
                for i in range(1024):
                    if len(index_str[i * 8:i * 8 + 8]) != 8:
                        break
                    index_tuple = st.unpack("q", index_str[i * 8:i * 8 + 8])
                    index_block.list.append(index_tuple[0])
                # bucket_id = index_block.list[pos_in_block]
                index_buffer.append(index_block)
    visit_time += math.ceil((nx * ny * nz) / 1024)
    index_buffer.block_dict.clear()
    index_buffer.id_list.clear()
    # 将bucket中大于x_mid对应的线性刻度的记录移到bucket_new中
    x_min = X[pos.x]
    x_max = X[pos.x + 1]
    z_min = Z[pos.z]
    z_max = Z[pos.z + 1]
    # 添加新桶
    new_bucket = Block()
    new_bucket.id = num
    new_bucket.list = []
    num += 1
    B.append(new_bucket)
    remove_record = []
    for record in B[old_bucket_id].list:
        if x_min <= record.x < x_max and Y[pos.y] <= record.y < mid_y and z_min <= record.z < z_max:  # 如果小于，则写入新桶，并删除
            B[new_bucket.id].list.append(record)
            remove_record.append(record)
    for record in remove_record:
        B[old_bucket_id].list.remove(record)
    # write_block(new_bucket)
    write_time += 1


def split_z(pos):
    """
    按z坐标分裂
    :param pos: grid array上的位置
    :return: 无
    """
    global X
    global Y
    global Z
    global B
    global index_file_name
    global num
    global nz
    global grid_array
    global visit_time
    global index_buffer
    global write_time
    old_bucket_id = grid_array[pos.x][pos.y][pos.z]
    mid_z = (Z[pos.z] + Z[pos.z + 1]) / 2
    Z.insert(pos.z + 1, mid_z)
    nz += 1
    # 给新分割的栅格块分配新桶
    # grid array切割
    grid_array = np.insert(grid_array, pos.z, grid_array[:, :, pos.z], axis=2)
    grid_array[pos.x][pos.y][pos.z + 1] = num
    for i in range(math.ceil((nx * ny * nz) / 1024)):
        # 现在缓冲区中查找索引块
        find_block = 0
        if len(index_buffer.id_list) != 0 and i in index_buffer.block_dict:
            block = index_buffer.block_dict[i]
            # bucket_id = block.list[pos_in_block]
            find_block = 1
        # 如果缓冲区中没找到，则去硬盘读取索引块
        if find_block == 0:
            with open(index_file_name, 'rb') as index_file:
                index_file.seek(i * 8 * 1024)
                index_str = index_file.read(8 * 1024)
                index_block = Block()
                index_block.id = i
                index_block.list = []
                for i in range(1024):
                    if len(index_str[i * 8:i * 8 + 8]) != 8:
                        break
                    index_tuple = st.unpack("q", index_str[i * 8:i * 8 + 8])
                    index_block.list.append(index_tuple[0])
                # bucket_id = index_block.list[pos_in_block]
                index_buffer.append(index_block)
    visit_time += math.ceil((nx * ny * nz) / 1024)
    index_buffer.block_dict.clear()
    index_buffer.id_list.clear()
    # 将bucket中大于x_mid对应的线性刻度的记录移到bucket_new中
    x_min = X[pos.x]
    x_max = X[pos.x + 1]
    y_min = Y[pos.y]
    y_max = Y[pos.y + 1]
    # 添加新桶
    new_bucket = Block()
    new_bucket.id = num
    new_bucket.list = []
    num += 1
    B.append(new_bucket)
    remove_record = []
    for record in B[old_bucket_id].list:
        if x_min <= record.x < x_max and y_min <= record.y < y_max and Z[pos.z] <= record.z < mid_z:  # 如果小于，则写入新桶，并删除
            B[new_bucket.id].list.append(record)
            remove_record.append(record)
    for record in remove_record:
        B[old_bucket_id].list.remove(record)
    # write_block(new_bucket)
    write_time += 1


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
    for i in range(nz):
        if r.z < Z[i]:
            pos.z = i - 1
            break
    return pos


def insert(r):
    """
    将记录r插入栅格文件中
    :param r: 点
    :return: 无
    """
    global flag     # 声明使用全局变量flag
    global count
    global ep
    global num
    global data_buffer
    global index_buffer
    global visit_time
    global write_time
    global data_file_name
    pos = find(r)
    if ep == 1:
        bucket_id = grid_array[pos.x][pos.y][pos.z]
        if len(B[bucket_id].list) <= max_c:
            B[bucket_id].list.append(r)
        elif len(B[bucket_id].list) > max_c:
            flag += 1
    elif ep > 1:
        bucket_id = grid_array[pos.x][pos.y][pos.z]
        B[bucket_id].list.append(r)
        # 根据栅格数组的值，找到桶
        # 根据grid array中的位置，计算出索引块号以及块中位置
        block_id = int((pos.x * ny * nz + pos.y * nz + pos.z) / 1024)
        pos_in_block = pos.x * ny * nz + pos.y * nz + pos.z - block_id * 1024
        # 现在缓冲区中查找索引块
        find_block = 0
        if len(index_buffer.id_list) != 0 and block_id in index_buffer.block_dict:
            block = index_buffer.block_dict[block_id]
            # bucket_id = block.list[pos_in_block]
            find_block = 1
        # 如果缓冲区中没找到，则去硬盘读取索引块
        if find_block == 0:
            with open(index_file_name, 'rb') as index_file:
                index_file.seek(block_id * 8 * 1024)
                index_str = index_file.read(8 * 1024)
                index_block = Block()
                index_block.id = block_id
                index_block.list = []
                for i in range(1024):
                    if len(index_str[i * 8:i * 8 + 8]) != 8:
                        break
                    index_tuple = st.unpack("q", index_str[i * 8:i * 8 + 8])
                    index_block.list.append(index_tuple[0])
                # bucket_id = index_block.list[pos_in_block]
                index_buffer.append(index_block)
        # 将记录r插入桶
        # 在缓存中寻找数据块，如果找到直接在缓存中的数据块中查找点
        find_block = 0
        if len(data_buffer.id_list) != 0 and bucket_id in data_buffer.block_dict:
            bucket = data_buffer.block_dict[bucket_id]
            bucket.list.append(r)
            # B[bucket_id].list.append(r)
            find_block = 1
        # 在缓存中没找到数据块，则到硬盘上找，并加入缓存中，然后查找点
        if find_block == 0:
            with open(data_file_name, 'rb') as data_file:
                data_file.seek(bucket_id * (max_c * 24 + 8))
                str = data_file.read(max_c * 24 + 8)
                bucket = Block()
                bucket.id = bucket_id
                bucket.list = []
                for i in range(max_c):
                    if len(str[i * 24:i * 24 + 24]) != 24:
                        break
                    record_str = st.unpack('ddd', str[i * 24:i * 24 + 24])
                    if record_str[0] == 0 and record_str[1] == 0 and record_str[2] == 0:
                        break
                    record = Record()
                    record.x = record_str[0]
                    record.y = record_str[1]
                    record.z = record_str[2]
                    bucket.list.append(record)
                bucket.list.append(r)
            # B[bucket_id].list.append(r)
        # 如果桶溢出,则进行分裂操作
        # 分裂前检查是否两个块指向一个桶！！！
        if len(B[bucket_id].list) > max_c:
            if grid_array[pos.x + 1][pos.y][pos.z] == bucket_id:
                remove_record = []
                new_bucket = Block()
                new_bucket.list = []
                new_bucket.id = num
                num += 1
                grid_array[pos.x + 1][pos.y][pos.z] = new_bucket.id
                write_time += 1
                for record in B[bucket_id].list:
                    if record.x >= X[pos.x + 1]:    # 如果大于，则写入新桶，并删除
                        new_bucket.list.append(record)
                        remove_record.append(record)
                for record in remove_record:
                    B[bucket_id].list.remove(record)
                B.append(new_bucket)
                # write_block(new_bucket)
                write_time += 1
            elif grid_array[pos.x - 1][pos.y][pos.z] == bucket_id:
                remove_record = []
                new_bucket = Block()
                new_bucket.list = []
                new_bucket.id = num
                num += 1
                grid_array[pos.x - 1][pos.y][pos.z] = new_bucket.id
                write_time += 1
                for record in B[bucket_id].list:
                    if record.x < X[pos.x]:  # 如果小于，则写入新桶，并删除
                        new_bucket.list.append(record)
                        remove_record.append(record)
                for record in remove_record:
                    B[bucket_id].list.remove(record)
                B.append(new_bucket)
                # write_block(new_bucket)
                write_time += 1
            elif grid_array[pos.x][pos.y + 1][pos.z] == bucket_id:
                remove_record = []
                new_bucket = Block()
                new_bucket.list = []
                new_bucket.id = num
                num += 1
                grid_array[pos.x][pos.y + 1][pos.z] = new_bucket.id
                write_time += 1
                for record in B[bucket_id].list:
                    if record.y >= Y[pos.y + 1]:  # 如果大于，则写入新桶，并删除
                        new_bucket.list.append(record)
                        remove_record.append(record)
                for record in remove_record:
                    B[bucket_id].list.remove(record)
                B.append(new_bucket)
                # write_block(new_bucket)
                write_time += 1
            elif grid_array[pos.x][pos.y - 1][pos.z] == bucket_id:
                remove_record = []
                new_bucket = Block()
                new_bucket.list = []
                new_bucket.id = num
                num += 1
                grid_array[pos.x][pos.y - 1][pos.z] = new_bucket.id
                write_time += 1
                for record in B[bucket_id].list:
                    if record.y < Y[pos.y]:  # 如果小于，则写入新桶，并删除
                        new_bucket.list.append(record)
                        remove_record.append(record)
                for record in remove_record:
                    B[bucket_id].list.remove(record)
                B.append(new_bucket)
                # write_block(new_bucket)
                write_time += 1
            elif grid_array[pos.x][pos.y][pos.z + 1] == bucket_id:
                remove_record = []
                new_bucket = Block()
                new_bucket.list = []
                new_bucket.id = num
                num += 1
                grid_array[pos.x][pos.y][pos.z + 1] = new_bucket.id
                write_time += 1
                for record in B[bucket_id].list:
                    if record.z >= Z[pos.z + 1]:  # 如果大于，则写入新桶，并删除
                        new_bucket.list.append(record)
                        remove_record.append(record)
                for record in remove_record:
                    B[bucket_id].list.remove(record)
                B.append(new_bucket)
                # write_block(new_bucket)
                write_time += 1
            elif grid_array[pos.x][pos.y][pos.z - 1] == bucket_id:
                remove_record = []
                new_bucket = Block()
                new_bucket.list = []
                new_bucket.id = num
                num += 1
                grid_array[pos.x][pos.y][pos.z - 1] = new_bucket.id
                write_time += 1
                for record in B[bucket_id].list:
                    if record.z < Z[pos.z]:  # 如果小于，则写入新桶，并删除
                        new_bucket.list.append(record)
                        remove_record.append(record)
                for record in remove_record:
                    B[bucket_id].list.remove(record)
                B.append(new_bucket)
                # write_block(new_bucket)
                write_time += 1
            # 设置一个flag 划分一次变一个值，循环划分，偶数横向划分，奇数纵向划分
            elif flag % 3 == 0:
                split_x(pos)
                flag += 1
            elif flag % 3 == 1:
                split_y(pos)
                flag += 1
            elif flag % 3 == 2:
                split_z(pos)
                flag += 1
        if find_block == 0:
            data_buffer.append(bucket)


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
    global nx, ny, nz
    # 根据linear scale找到grid array中的位置
    pos = find(r)
    # 根据grid array中的位置，计算出索引块号以及块中位置
    block_id = int((pos.x * ny * nz + pos.y * nz + pos.z) / 1024)
    pos_in_block = pos.x * ny * nz + pos.y * nz + pos.z - block_id * 1024
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
            if record.x == r.x and record.y == r.y and record.z == r.z:
                return 1
        return 0
    # 在缓存中没找到数据块，则到硬盘上找，并加入缓存中，然后查找点
    with open(data_file_name, 'rb') as data_file:
        visit_time += 1
        # data_file.seek(0)
        # for i in range(bucket_id - 1):
        #     data_file.seek(max_c * 24 + 8, 1)  # 每个桶存max_c*24B的数据，即max_c个点
        data_file.seek(bucket_id * (max_c * 24 + 8))
        str = data_file.read(max_c * 24 + 8)
        bucket = Block()
        bucket.id = bucket_id
        bucket.list = []
        for i in range(max_c):
            if len(str[i * 24:i * 24 + 24]) != 24:
                break
            record_str = st.unpack('ddd', str[i*24:i*24+24])
            if record_str[0] == 0 and record_str[1] == 0 and record_str[2] == 0:
                break
            record = Record()
            record.x = record_str[0]
            record.y = record_str[1]
            record.z = record_str[2]
            bucket.list.append(record)
        data_buffer.append(bucket)
        for record in bucket.list:
            if record.x == r.x and record.y == r.y and record.z == r.z:
                return 1
    return 0


def range_query(rang):
    """
    范围查询
    :param rang: 范围
    :return: 找到的点数组
    """
    global visit_time
    global index_buffer
    global data_buffer
    global bucket_count
    bucket_id_set = set()
    pos_list = []
    range_array = []
    # grid array中结束点的位置
    r = Record()
    r.x = rang.x_max
    r.y = rang.y_max
    r.z = rang.z_max
    end_pos = find(r)
    # grid array中开始点的位置
    r = Record()
    r.x = rang.x_min
    r.y = rang.y_min
    r.z = rang.z_min
    start_pos = find(r)
    # 将范围中每个点的位置存入位置数组
    for i in range(start_pos.x, end_pos.x + 1):
        for j in range(start_pos.y, end_pos.y + 1):
            for k in range(start_pos.z, end_pos.z + 1):
                pos = Position()
                pos.x = i
                pos.y = j
                pos.z = k
                pos_list.append(pos)
    # 遍历位置数组，在索引块中查找并加入桶号集合
    for pos in pos_list:
        # 根据grid array中的位置，计算出索引块号以及块中位置
        block_id = int((pos.x * ny * nz + pos.y * nz + pos.z) / 1024)
        pos_in_block = pos.x * ny * nz + pos.y * nz + pos.z - block_id * 1024
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
                if (rang.x_min <= record.x <= rang.x_max) and (rang.y_min <= record.y <= rang.y_max) and (
                        rang.z_min <= record.z <= rang.z_max):
                    range_array.append(record)
        # 在缓存中没找到数据块，则到硬盘上找，并加入缓存中，然后查找点
        if find_block == 0:
            with open(data_file_name, 'rb') as data_file:
                visit_time += 1
                # data_file.seek(0)
                # for i in range(bucket_id - 1):
                #     data_file.seek(max_c * 24 + 8, 1)  # 每个桶存max_c*24B的数据，即max_c个点
                data_file.seek(bucket_id * (max_c * 24 + 8))
                str = data_file.read(max_c * 24 + 8)
                bucket = Block()
                bucket.id = bucket_id
                bucket.list = []
                data_buffer.append(bucket)
                for i in range(max_c):
                    record_str = st.unpack('ddd', str[i * 24:i * 24 + 24])
                    if record_str[0] == 0 and record_str[1] == 0 and record_str[2] == 0:
                        break
                    record = Record()
                    record.x = record_str[0]
                    record.y = record_str[1]
                    record.z = record_str[2]
                    bucket.list.append(record)
                    if (rang.x_min <= record.x <= rang.x_max) and (rang.y_min <= record.y <= rang.y_max) and (
                            rang.z_min <= record.z <= rang.z_max):
                        range_array.append(record)
    return range_array


DATA = "SIM_DATA"
max_c = 341  # 25 * 7   # 最大桶容量 10<c<1000 c=70效果最好 7 70 700
flag = 0  # 表示分裂次数，用来循环分裂操作flag % 3 == 0时分裂x；flag % 3 == 1时分裂y；flag % 3 == 2时分裂z

data_file_name = "../SIM_DATA/sim.dat"
index_file_name = "../SIM_DATA/sim.idx"
range_query_file_name = "../SIM_DATA/rangeQuery_sim.txt"
point_query_file1_name = "../SIM_DATA/pointQuery1_sim.txt"
point_query_file2_name = "../SIM_DATA/pointQuery2_sim.txt"
nx = 29  #
ny = 29  #
nz = 29  # grid array 大小
max_x_value = 50000  # x
min_x_value = 0
max_y_value = 50000  # y
min_y_value = 0
max_z_value = 1000  # t
min_z_value = 0
max_count = 300000  # 一期30w数据
ep_start_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ep_time = 1000

# 建立模型
# 建立3维栅格数组
# 3维栅格数组，数组值为桶号，初始为桶0
grid_array = np.arange(nx * ny * nz, dtype='int64').reshape(nx, ny, nz)
# 建立线性刻度，方便查找
X = [min_x_value]  # 线性刻度
Y = [min_y_value]
Z = [min_z_value]
# 建立B数组，用来装桶
B = []
num = 0
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
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
index_buffer.max_size = 640
index_buffer.id_list = []
index_buffer.block_dict = {}
visit_time = 0
write_time = 0
count = 0
R = []
ep = 1
print('ep' + str(ep))
time_start = time.time()
print(DATA)
for i in range(300):
    with open("../" + DATA + '/EP1/' + str(i) + '.txt') as file_object:
        for line in file_object:
            line = line.rstrip()
            line = line.split(",")
            r = Record()
            r.x = float(line[3])
            r.y = float(line[4])
            r.z = int(line[1])
            R.append(r)
visit_time += math.ceil(max_count / 341)
print(len(R))
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
R.sort(key=attrgetter("z"))
for i in range(1, nz):
    Z.append(R[i * int(max_count / nz)].z)
Z.append(max_z_value)
print(Z)
for r in R:
    insert(r)
print('flag = ' + str(flag))
# 将grid array 写入硬盘
index_file = open(index_file_name, 'wb')
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            bucket_id = st.pack("q", grid_array[i][j][k])
            index_file.write(bucket_id)
visit_time += math.ceil((nx * ny * nz) / 1024)
index_file.close()
# 将桶中数据写入硬盘
data_file = open(data_file_name, 'wb')
write_time_start = time.time()
for i in range(num):
    write_block(B[i])
    visit_time += 1
write_time_end = time.time()
data_file.close()
time_end = time.time()
print('cost of build time:', time_end - time_start, 's')
print('visit time:', visit_time)
print(visit_time * 0.3, "ms")
print("\n")

# 静态装入点查询
# 一个索引块可以放1024(1000)个索引项 1000 * 8
# 一个数据块可以放341个数据
print("point query")
data_buffer = Buffer()
data_buffer.max_size = 128
data_buffer.id_list = []
data_buffer.block_dict = {}
index_buffer = Buffer()
index_buffer.max_size = 640
index_buffer.id_list = []
index_buffer.block_dict = {}
time_start = time.time()
success = 0
fail = 0
visit_time = 0
write_time = 0
index_file = open(index_file_name, "rb")
data_file = open(data_file_name, "rb")
with open(point_query_file1_name) as file_object:
    for line in file_object:
        line = line.rstrip()
        line = line.split(",")
        r = Record()
        r.x = float(line[0])
        r.y = float(line[1])
        r.z = float(line[2])
        result = point_query(r)
        if result == 1:
            success += 1
        elif result == 0:
            fail += 1
visit_time += math.ceil(500 / 341)
time_end = time.time()
print('point_query_time cost:', time_end - time_start, 's')
print('visit time:', visit_time)
print("\n")

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
write_time = 0
bucket_count = 0
time_start = time.time()
# 将索引块加入缓存区
index_file = open(index_file_name, "rb")
data_file = open(data_file_name, "rb")
with open(range_query_file_name) as file_object:
    for line in file_object:
        line = line.rstrip()
        line = line.split(",")
        rang = Range()
        rang.x_min = float(line[0])
        rang.x_max = float(line[1])
        rang.y_min = float(line[2])
        rang.y_max = float(line[3])
        rang.z_min = float(line[4])
        rang.z_max = float(line[5])
        if rang.z_max > ep_time:
            rang.z_max = ep_time
        result = range_query(rang)
        record_count = len(result)
        record_sum += record_count
visit_time += math.ceil(1000 / 341)
time_end = time.time()
print('range_query_time cost:', time_end - time_start, 's')
print('visit time:', visit_time)
print('record sum:', record_sum)
print('bucket count:', bucket_count)
print("\n")

# 动态装入 ep2~ep10
flag = 0
for ep in range(2, 11):
    index_file = open(index_file_name, "rb")
    data_file = open(data_file_name, "rb")
    visit_time = 0
    write_time = 0
    data_buffer = Buffer()
    data_buffer.max_size = 128
    data_buffer.id_list = []
    data_buffer.block_dict = {}
    index_buffer = Buffer()
    index_buffer.max_size = 640
    index_buffer.id_list = []
    index_buffer.block_dict = {}
    print('ep' + str(ep))
    time_start = time.time()
    for i in range(300):
        with open("../" + DATA + '/EP' + str(ep) + '/' + str(i) + '.txt') as file_object:
            for line in file_object:
                line = line.rstrip()
                line = line.split(",")
                r = Record()
                r.x = float(line[3])
                r.y = float(line[4])
                r.z = int(line[1])
                insert(r)
    visit_time += math.ceil(max_count / 341)
    index_file.close()
    data_file.close()
    # 将grid array 写入硬盘
    index_file = open(index_file_name, 'wb')
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                bucket_id = st.pack("q", grid_array[i][j][k])
                index_file.write(bucket_id)
    index_file.close()
    # 将桶中数据写入硬盘
    data_file = open(data_file_name, 'wb')
    for i in range(num):
        write_block(B[i])
    data_file.close()
    time_end = time.time()
    maintain_time = time_end - time_start + ((visit_time + write_time) * 0.3 / 1000)
    print('flag = ' + str(flag))
    print('nx = ' + str(nx))
    print('ny = ' + str(ny))
    print('nz = ' + str(nz))
    print('x = ' + str(len(X)))
    print('y = ' + str(len(Y)))
    print('z = ' + str(len(Z)))
    print(X)
    print(Y)
    print(Z)
    print("visit time:", visit_time)
    print("write time:", write_time)
    print("maintain time:", maintain_time, 's')
    print("\n")

# 周期更新点查询
# 一个索引块可以放1024(1000)个索引项 1000 * 8
# 一个数据块可以放341个数据
print("point query")
data_buffer = Buffer()
data_buffer.max_size = 128
data_buffer.id_list = []
data_buffer.block_dict = {}
index_buffer = Buffer()
index_buffer.max_size = 640
index_buffer.id_list = []
index_buffer.block_dict = {}
time_start = time.time()
success = 0
fail = 0
visit_time = 0
write_time = 0
read_time_count = 0
read_count = 0
index_file = open(index_file_name, "rb")
data_file = open(data_file_name, "rb")
with open(point_query_file2_name) as file_object:
    for line in file_object:
        line = line.rstrip()
        line = line.split(",")
        r = Record()
        r.x = float(line[0])
        r.y = float(line[1])
        r.z = float(line[2]) % ep_time
        result = point_query(r)
        if result == 1:
            success += 1
        elif result == 0:
            fail += 1
visit_time += math.ceil(500 / 341)
time_end = time.time()
print('point_query_time cost:', time_end - time_start, 's')
print('visit time:', visit_time)
print("\n")

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
write_time = 0
bucket_count = 0
time_start = time.time()
# 将索引块加入缓存区
index_file = open(index_file_name, "rb")
data_file = open(data_file_name, "rb")
with open(range_query_file_name) as file_object:
    for line in file_object:
        line = line.rstrip()
        line = line.split(",")
        rang = Range()
        rang.x_min = float(line[0])
        rang.x_max = float(line[1])
        rang.y_min = float(line[2])
        rang.y_max = float(line[3])
        rang.z_min = float(line[4])
        rang.z_max = float(line[5])
        if rang.z_max > ep_time:
            rang.z_max = ep_time
        result = range_query(rang)
        record_count = len(result)
        record_sum += record_count
visit_time += math.ceil(1000 / 341)
time_end = time.time()
print('range_query_time cost:', time_end - time_start, 's')
print('visit time:', visit_time)
print('record sum:', record_sum)
print('bucket count:', bucket_count)