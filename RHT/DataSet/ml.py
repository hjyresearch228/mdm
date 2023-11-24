import os
import time
import tensorflow as tf
from sklearn import linear_model
import struct as st
import DataSet.hilbertFromGithub as hb
import math as mt
import numpy as np

class Point:
    max_pos = -1.0
    min_pos = float('inf')
    min_time=float('inf')
    max_time=0
    def __init__(self,pos,time):
        self.pos=pos  # pos维
        self.time=time  # time维
        self.par_id=0  # 分区号
        self.dist = 0.0  # 到参考点的距离
        self.key=0.0  # 一维值
        self.addr=0  # 地址

class Partition:
    def __init__(self):
        self.id=0  # 分区号
        self.radii=0.0  # 分区半径
        self.ref=None  # 参考点
        self.offset=0.0  # 偏移量

class DataBlock:
    DataMaxBlock = 0
    BlockSize = 512

class buffer:
    '''
    缓存空间设置
    '''
    InsertTest = []
    InsertId = []
    InsertBlockNum = 0

class P:
    Dat='ml_sim.dat'


def ReadBlockByIdForInsert(id):
    '''
    增量插入时，用的读取快
    :param id:
    :return:
    '''
    tempfile = open(P.Dat, 'r+b')
    tempfile.seek(id * 8192)
    pointList = []
    tempStr = tempfile.read(8192)
    for i in range(DataBlock.BlockSize):
        mm = st.unpack('dd', tempStr[i * 16:i * 16 + 16])
        p=Point(mm[0],mm[1])
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


def getDistance(p, q):
    return mt.sqrt((p.pos - q.pos) ** 2 + (p.time - q.time) ** 2)


def scaling_method(point):
    '''
    放缩方法，把多维数据映射到一维
    :param point: 多维数据点
    :return: 一维数据
    '''
    pos_l = (Point.max_pos - Point.min_pos)/8
    time_l = (Point.max_time - Point.min_time)/8
    partitions = {}  # 分区
    for i in range(8):
        for j in range(8):
            par=Partition()
            par.id = hb.hilbert_index(2, 3, [i, j])
            ref_pos=(2*Point.min_pos+j*pos_l+(j+1)*pos_l)/2
            ref_time=(2*Point.min_time+i*time_l+(i+1)*time_l)/2
            par.ref=Point(ref_pos,ref_time)
            partitions[par.id]=par
    for p in point:  # 获取点的分区号和分区半径
        row=int(p.time//time_l)
        if row>7:
            row=7
        col=int(p.pos//pos_l)
        if col>7:
            col=7
        p.par_id=hb.hilbert_index(2, 3, [row, col])
        r=getDistance(p,partitions[p.par_id].ref)
        p.dist=r
        if r>partitions[p.par_id].radii:
            partitions[p.par_id].radii=r
    point_key={}
    for j in range(len(point)):
        offset=0
        for i in range(point[j].par_id):
            offset+=partitions[i].radii
        partitions[point[j].par_id].offset=offset
        point[j].key=point[j].dist+offset
        point_key[j]=point[j].key
    point_order=sorted(point_key.items(),key=lambda x:x[1])  # 按一维值排序
    order=0
    for tup in point_order:
        point[tup[0]].addr=order
        order+=1
    return point,partitions


def scaling_method_real(point):
    '''
    放缩方法，把多维数据映射到一维
    :param point: 多维数据点
    :return: 一维数据
    '''
    pos_l = (Point.max_pos - Point.min_pos)/8
    time_l = (Point.max_time - Point.min_time)/8
    partitions = {}  # 分区
    for i in range(16):
        for j in range(16):
            par=Partition()
            par.id = hb.hilbert_index(2, 4, [i, j])
            ref_pos=(2*Point.min_pos+j*pos_l+(j+1)*pos_l)/2
            ref_time=(2*Point.min_time+i*time_l+(i+1)*time_l)/2
            par.ref=Point(ref_pos,ref_time)
            partitions[par.id]=par
    for p in point:  # 获取点的分区号和分区半径
        row=int(p.time//time_l)
        if row>15:
            row=15
        col=int(p.pos//pos_l)
        if col>15:
            col=15
        p.par_id=hb.hilbert_index(2, 4, [row, col])
        r=getDistance(p,partitions[p.par_id].ref)
        p.dist=r
        if r>partitions[p.par_id].radii:
            partitions[p.par_id].radii=r
    point_key={}
    for j in range(len(point)):
        offset=0
        for i in range(point[j].par_id):
            offset+=partitions[i].radii
        partitions[point[j].par_id].offset = offset
        point[j].key=point[j].dist+offset
        point_key[j]=point[j].key
    point_order=sorted(point_key.items(),key=lambda x:x[1])  # 按一维值排序
    order=0
    for tup in point_order:
        point[tup[0]].addr=order
        order+=1
    return point,partitions


def  build_model():
    '''
    :return:神经网络模型
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, input_shape=(1,), activation='relu'))
    model.add(tf.keras.layers.Dense(16,activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


def read_dataset(path, datasets):
    '''
    读取数据集
    :param path: 数据存放的路径
    :param datasets: 存放读取到的轨迹点
    '''
    dirs = os.listdir(path)
    tempData = []  # 从文件中读取到的数据
    for file in dirs:
        s = path + '\\' + file
        fo = open(s)
        line = fo.readline()
        while line:
            tempData.append(line)
            line = fo.readline()
        fo.close()
    splitData = []
    for i in range(len(tempData)):
        splitData.append(tempData[i].split(","))
    for j in range(len(splitData)):
        p=Point(float(splitData[j][3]),float(splitData[j][4]))
        datasets.append(p)
        if float(splitData[j][3])>Point.max_pos:
            Point.max_pos=float(splitData[j][3])
        if float(splitData[j][3])<Point.min_pos:
            Point.min_pos=float(splitData[j][3])
        if float(splitData[j][4])>Point.max_time:
            Point.max_time=float(splitData[j][4])
        if float(splitData[j][4])<Point.min_time:
            Point.min_time=float(splitData[j][4])


def range_query(q,nn,model_list,eb,partition,point_num):
    '''
    范围查询
    :param q: 范围,[pos_min,pos_max,time_min,time_max]
    :param nn: 第一阶段模型
    :param model_list: 第二阶段模型
    :param eb: 误差
    :param partition: 分区
    :param point_num: 数据点个数
    '''
    block_set=set()  # 预测的位置
    for i in range(len(partition)):  # 对每个分区进行查询
        if partition[i].radii == 0.0:
            continue
        ref_point=partition[i].ref  # 参考点
        if ref_point.pos<=q[0]:
            closest_pos=q[0]
            farthest_pos=q[1]
        if ref_point.pos>=q[1]:
            closest_pos = q[1]
            farthest_pos = q[0]
        if ref_point.pos>q[0] and ref_point.pos<q[1]:
            closest_pos = ref_point.pos
            if abs(q[0]-ref_point.pos)>=abs(q[1]-ref_point.pos):
                farthest_pos = q[0]
            else:
                farthest_pos = q[1]
        if ref_point.time<=q[2]:
            closest_time=q[2]
            farthest_time=q[3]
        if ref_point.time>=q[3]:
            closest_time = q[3]
            farthest_time = q[2]
        if ref_point.time>q[2] and ref_point.time<q[3]:
            closest_time = ref_point.time
            if abs(q[2]-ref_point.time)>=abs(q[3]-ref_point.time):
                farthest_time = q[2]
            else:
                farthest_time = q[3]
        closest_point=Point(closest_pos,closest_time)  # 最近的点
        farthest_point=Point(farthest_pos,farthest_time)  # 最远的点
        min_dist=getDistance(ref_point,closest_point)
        max_dist=getDistance(ref_point,farthest_point)
        if min_dist>partition[i].radii:
            continue
        if max_dist>partition[i].radii:
            max_dist=partition[i].radii
        offs1 = min_dist + partition[i].offset
        offs2 = max_dist + partition[i].offset
        pre0 = nn.predict(np.array([offs1]).reshape(-1, 1))
        num0 = mt.floor(pre0[0] / point_num * len(model_list))
        if num0<0:
            num0=0
        if num0>len(model_list)-1:
            num0 = len(model_list)-1
        pred0=model_list[num0].predict(np.array([offs1]).reshape(-1, 1))
        addr0=pred0[0]-eb[num0]*0.6
        if addr0<0:
            addr0=0
        pre1 = nn.predict(np.array([offs2]).reshape(-1, 1))
        num1 = mt.floor(pre1[0] / point_num * len(model_list))
        if num1 < 0:
            num1 = 0
        if num1 > len(model_list) - 1:
            num1 = len(model_list) - 1
        pred1 = model_list[num1].predict(np.array([offs2]).reshape(-1, 1))
        addr1 = pred1[0] + eb[num1]*0.6
        if addr1 >= point_num:
            addr1 = point_num-1
        for addrs in range(int(addr0),int(addr1+1)):
            block_id=addrs//DataBlock.BlockSize
            block_set.add(block_id)
    return block_set


def read_range(file_name):
    range_query = []
    tempData=[]
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
        pos_min = float(splitData[j][0])
        pos_max = float(splitData[j][1])
        time_min = float(splitData[j][2])
        time_max=float(splitData[j][3])
        range_query.append([pos_min,pos_max,time_min,time_max])
    return range_query


def range_filter(q,block_id):
    res_num=0
    for id in block_id:
        point_list=IsBlockInBuffer(id)
        for p in point_list:
            if p.pos>=q[0] and p.pos<=q[1] and p.time>=q[2] and p.time<=q[3]:
                res_num+=1
    return res_num


def write_data(file_name,point):
    '''
    将数据点的二维坐标写入指定文件中
    :param file_name:文件名
    :param points: 字典类型，键为希尔伯特值，值为数据点
    :return:
    '''
    point_key = {}
    for j in range(len(point)):
        point_key[point[j].addr] = point[j]
    point_order = sorted(point_key.items(), key=lambda x: x[0])
    file = open(file_name, 'wb')
    block_num = mt.ceil(len(point) / DataBlock.BlockSize)  # 块数
    DataBlock.DataMaxBlock=block_num-1
    block_id=0
    while block_id<block_num:
        for i in range(block_id * DataBlock.BlockSize, (block_id + 1) * DataBlock.BlockSize):
            if i < len(point):
                p = point_order[i][1]
                b2 = st.pack('d', p.pos)
                b3 = st.pack('d', p.time)
                file.write(b2)
                file.write(b3)
            else:
                b2 = st.pack('d', 0)
                b3 = st.pack('d', 0)
                file.write(b2)
                file.write(b3)
        block_id+=1
    file.close()


def ml_sim(Period=1):
    point=[]  # 数据点
    path="..//SIM//EP"+str(Period)
    read_dataset(path,point)
    point,partitions=scaling_method(point)
    write_data(P.Dat, point)  # 写数据
    x_test=[]
    y_test=[]
    for p in point:
        x_test.append(p.key)
        y_test.append(p.addr)
    nn_model=build_model()
    nn_model.compile(optimizer='adam', loss='mse')
    t1 = time.time()
    nn_model.fit(np.array(x_test).reshape(-1, 1), np.array(y_test), epochs=20)
    t2 = time.time()
    print("构建时间：", t2 - t1)
    pre = nn_model.predict(np.array(x_test).reshape(-1, 1))
    M1=10  # 第二阶段模型个数
    X=[]
    Y=[]
    for i in range(M1):
        X.append([])
        Y.append([])
    for i in range(len(point)):
        num = mt.floor(pre[i] / len(point) * M1)  # 第几个模型
        if num<0:
            num=0
        if num>M1-1:
            num=M1-1
        X[num].append(point[i].key)
        Y[num].append(point[i].addr)
    model_list={}  # 第二阶段模型
    max_err=[]  # 模型最大误差
    for i in range(M1):
        max_err.append(0)
        x_train = X[i]
        y_train = Y[i]
        if len(x_train)>0:
            lr = linear_model.LinearRegression()  # 线性回归模型
            lr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train))
            model_list[i]=lr
            pred=lr.predict(np.array(x_train).reshape(-1, 1))
            for j in range(len(y_train)):
                err=abs(y_train[j]-pred[j])
                if err>max_err[i]:
                    max_err[i]=err
    ranges = read_range("..//SIM_DATA//range_sim.txt")  # 读取范围
    res_num=0  # 查询到的点数
    t3=time.time()
    for q in ranges:
        block_id=range_query(q,nn_model,model_list,max_err,partitions,len(point))
        num=range_filter(q,block_id)
        res_num+=num
    t4=time.time()
    print("找到的数据点数：", res_num)
    print("查找耗时：",t4-t3)
    real_num = 0
    for qr in ranges:
        for p in point:
            if p.pos >= qr[0] and p.pos <= qr[1] and p.time >= qr[2] and p.time <= qr[3]:
                real_num += 1
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：", buffer.InsertBlockNum)


def ml_geo(Period=1):
    point=[]  # 数据点
    path="..//GEO//EP"+str(Period)
    read_dataset(path,point)
    point,partitions=scaling_method_real(point)
    P.Dat='ml_geo.dat'
    write_data(P.Dat, point)  # 写数据
    x_test=[]
    y_test=[]
    for p in point:
        x_test.append(p.key)
        y_test.append(p.addr)
    nn_model=build_model()
    nn_model.compile(optimizer='adam', loss='mse')
    t1 = time.time()
    nn_model.fit(np.array(x_test).reshape(-1, 1), np.array(y_test), epochs=30)
    t2 = time.time()
    print("构建时间：", t2 - t1)
    pre = nn_model.predict(np.array(x_test).reshape(-1, 1))
    M1=10  # 第二阶段模型个数
    X=[]
    Y=[]
    for i in range(M1):
        X.append([])
        Y.append([])
    for i in range(len(point)):
        num = mt.floor(pre[i] / len(point) * M1)  # 第几个模型
        if num<0:
            num=0
        if num>M1-1:
            num=M1-1
        X[num].append(point[i].key)
        Y[num].append(point[i].addr)
    model_list={}  # 第二阶段模型
    max_err = []  # 模型最大误差
    for i in range(M1):
        max_err.append(0)
        x_train = X[i]
        y_train = Y[i]
        if len(x_train) > 0:
            lr = linear_model.LinearRegression()  # 线性回归模型
            lr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train))
            model_list[i] = lr
            pred = lr.predict(np.array(x_train).reshape(-1, 1))
            for j in range(len(y_train)):
                err = abs(y_train[j] - pred[j])
                if err > max_err[i]:
                    max_err[i] = err
    ranges = read_range("..//GeoData//range_geo.txt")  # 读取范围
    res_num = 0  # 查询到的点数
    t3 = time.time()
    for q in ranges:
        block_id = range_query(q, nn_model, model_list, max_err, partitions, len(point))
        num = range_filter(q, block_id)
        res_num += num
    t4 = time.time()
    print("找到的数据点数：", res_num)
    print("查找耗时：", t4 - t3)
    real_num = 0
    for qr in ranges:
        for p in point:
            if p.pos >= qr[0] and p.pos <= qr[1] and p.time >= qr[2] and p.time <= qr[3]:
                real_num += 1
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：", buffer.InsertBlockNum)


if __name__ == "__main__":
    # 模拟数据测试
    ml_sim(Period=1)
    # 真实数据测试
    # ml_geo(Period=1)