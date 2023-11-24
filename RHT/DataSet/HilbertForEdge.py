import math as mt
import math
import os
import time
import struct as st
import numpy as np
import DataSet.hilbertFromGithub  as hbg
from DataSet.rht import CART
from sklearn import tree
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import tensorflow as tf


class Point:
    # 用来存储当前路段中间的边界
    maxX = -1
    maxY = -1
    minX = float('inf')
    minY = float('inf')
    maxHilbert = 0
    minHilbert = float('inf')
    maxTime = 0  #所有数据点t的最大值
    minTime = float('inf')  #所有数据点t的最小值
    #pos的边界
    max_pos = -1.0
    min_pos = float('inf')

    # 轨迹点的定义方式
    def __init__(self, x, y):
        '''
        :param x: 经度
        :param y: 纬度
        '''
        # x为经度，y为纬度,希尔伯特值和时间戳默认为-1
        self.x = x
        self.y = y
        self.time = None
        self.hilbert = None
        self.hilbert_id=None
        self.traId = None
        self.SortId = None
        self.edgeId = None
        self.nearEdge = set()
        self.pos=None #segmentId.pos

    def __eq__(self, other):
        '''
        判断两个对象是否相同
        :param other: 要比较的对象
        :return: Bool类型
        '''
        return self.x == other.x and self.y == other.y and self.time == other.time

    def __hash__(self):
        return hash(self.x) * hash(self.y) + hash(self.time)

    def __lt__(self, other):
        if self.x < other.x:
            return True
        elif self.x == other.x:
            if self.y <= other.y:
                return True
            else:
                return False
        else:
            return False

    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.time)


# 路段，由两个交叉点组成
class Edge:
    def __init__(self, id, start, end, length):
        # 属性分别为 当前路段id，起始点id，终止点id，当前路段长度
        """
        :param id: 路段id
        :param start: 起始点
        :param end: 终止点
        :param length: 长度
        :param hilbert:希尔伯特空间的位置
        :param isTwo: 是否是重复路段
        """
        self.id = id
        self.start = start
        self.end = end
        self.length = length
        self.hilbert = None
        self.isTwo = 0
        self.nearEdge = []
        self.allNear = set()
        firstset = set()
        self.nearEdge.append(firstset)

    def __str__(self):
        return "此时的路段的id==" + str(self.id) + " \n当前开始和结束:" + str(self.start) + "   " + str(self.end)

    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end)

    def __hash__(self):
        return self.start * self.end

#  二维坐标样本点
class Sample:
    def __init__(self,pos,time):
        self.pos=pos
        self.time=time
        self.x=None
        self.y=None
        self.id=None
        self.hilbert=None
        self.addr=None

    def __eq__(self, other):
        '''
        判断两个对象是否相同
        :param other: 要比较的对象
        :return: Bool类型
        '''
        return self.pos == other.pos and self.time == other.time

    def __hash__(self):
        return hash(self.pos) * hash(self.time)


class P:
    Expend = 1
    Dat = 'Period1.dat'
    # 以下参数boost中会用到
    FirstSuccessNum = 0
    SecondSuccessNum=0
    ThirdSuccessNum=0
    OutDataNum=0


class DataBlock:
    DataMaxBlock = 0  # 数据存储的块数
    OutDataBlockSize = 0  # 溢出块的个数
    BlockSize = 512  # 块容量


# 溢出块类
class OutData:
    Start = -1
    index = -1


class buffer:
    '''
    缓存空间设置
    '''
    IndexBuffer = []
    DataIdBuffer = []
    DataBuffer = {}
    InsertTest = []
    InsertId = []
    InsertBlockNum = 0
    MapMaxData = {}


def getDistance1(p, q):
    return mt.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2)


def get_Map_From_Native(PointStr, EdgeStr):
    '''
    从本地获取地图信息，输入是点的存放地址和路段的存放地址，输出为路段集合和点的集合
    '''
    file_of_point = open(PointStr)
    file_of_edge = open(EdgeStr)
    list_Ver = []
    list_Edge = []
    for line in file_of_point:
        tempList = line.split(',')
        id1 = int(tempList[0])
        x = float(tempList[1])
        y = float(tempList[2])
        junction = Point(x, y)
        junction.id = id1
        list_Ver.append(junction)
    ver_list = list_Ver
    for line in file_of_edge:
        tempList = line.split(',')
        id1 = int(tempList[0])
        start = int(tempList[1])
        end = int(tempList[2])
        length = getDistance1(ver_list[start], ver_list[end])
        cur_edge = Edge(id1, start, end, length)
        list_Edge.append(cur_edge)
        list_Ver[start].nearEdge.add(id1)
        list_Ver[end].nearEdge.add(id1)
    return list_Ver, list_Edge


def get_middle_point(list_edge, list_ver):
    '''
    获取每个路段的中点
    :param list_edge: 路段集合
    :param list_ver: 点的集合
    :return: 路段集合中路段的中点的集合
    '''
    middle_point_list = []
    for edge in list_edge:
        x = (list_ver[edge.start].x + list_ver[edge.end].x) / 2
        y = (list_ver[edge.start].y + list_ver[edge.end].y) / 2
        if x > Point.maxX:
            Point.maxX = x
        if x < Point.minX:
            Point.minX = x
        if y > Point.maxY:
            Point.maxY = y
        if y < Point.minY:
            Point.minY = y
        tempPoint = Point(x, y)
        tempPoint.edgeId = edge.id
        middle_point_list.append(tempPoint)
    return middle_point_list


def get_hilbert_N(pointlist, edge_list):
    '''
    为每个路段赋予一个独立的希尔伯特值
    :param pointlist:地图顶点集合
    :param edge_list:路段集合
    :return:阶数，存放排序后的希尔伯特值和边的元组的列表
    '''
    # 建立一个字典，每个希尔伯特是否对应单个值.
    n = 15
    mm = 1
    while (mm == 1):
        #print("当前的阶数为" + str(n))
        p_dic = {}
        interX = (Point.maxX - Point.minX) / (1 << n)
        interY = (Point.maxY - Point.minY) / (1 << n)
        count1 = 0
        # 为每个路段中点赋予相应的希尔伯特值
        for i in range(len(pointlist)):
            x = 0
            if (pointlist[i].x - Point.minX) / interX != 0:
                x = mt.ceil((pointlist[i].x - Point.minX) / interX)
            y = 0
            if (pointlist[i].y - Point.minY) / interY != 0:
                y = mt.ceil((pointlist[i].y - Point.minY) / interY)
            hilbert = hbg.hilbert_index(2, n, [x, y])

            while (hilbert in p_dic.keys()):
                hilbert = hilbert + 1
                count1 += 1
            if edge_list[i].isTwo == 1:
                tempNum = 1
                while hilbert + tempNum in p_dic.keys():
                    tempNum += 1
                p_dic[hilbert + tempNum] = edge_list[i]
            else:
                p_dic[hilbert] = edge_list[i]

            if i == len(pointlist) - 1:
                mm = 0
    d_order = sorted(p_dic.items(), key=lambda x: x[0], reverse=False)  # 按字典集合中，每一个元组的键值元素排列。
    return n, d_order


def get_one_dimension_edge(sortEdge):
    '''
    将经过排序后的路段转化为一维空间上间隔，同时将每个路段附上映射的一维空间上的位置，第一个输入是经过排序后的路段顺序，输出是路段的一维空间表示
    :param sortEdge: 排序后的路段
    :return:
    '''
    id = 0
    for edge in sortEdge:
        edge[1].hilbertId = id
        edge[1].hilbert = edge[0]
        id += 1


def get_Data_From_Trajectory(str1, trajID):
    '''
    从本地获取轨迹数据，输出为轨迹点的集合
    :param str1: 轨迹的id
    :param trajID: 轨迹的坐标
    :return:
    '''
    file = open(str1)
    Point_list = []
    for line in file:
        TraList = line.split(',')
        x = float(TraList[3])
        y = float(TraList[4])
        time = int(TraList[1])
        if time > 2000:
            break
        id = int(TraList[2]) + 1
        Traid = int(TraList[0])
        tempPoint = Point(x, y)
        tempPoint.traId = Traid
        tempPoint.time = time
        if time > Point.maxTime:
            Point.maxTime = time
        if time < Point.minTime:
            Point.minTime = time
        tempPoint.traId = trajID
        tempPoint.edgeId = id
        Point_list.append(tempPoint)
    return Point_list


def get_Alldata_From_Native(start, end,Period=1):
    '''
    从本地获取所有的轨迹点，输出所有轨迹的集合
    :param start: 起始轨迹点的id
    :param end: 轨迹的坐标
    :return:
    '''
    allPoint = []
    for i in range(start, end):
        allPoint.extend(get_Data_From_Trajectory("..//SIM_DATA//EP"+str(Period)+"//" + str(i) + ".txt", i))
    return allPoint


def remove_Same_Point(PointAll):
    '''
    对列表进行去重操作
    :param PointAll:
    :return:
    '''
    new_list = list(set(PointAll))
    return new_list


def give_Point_hilbert(edgeList, verList, start, end,Period=1):
    '''
    求数据点的pos值
    :param edgeList: 路段集合
    :param verList: 顶点集合
    :param start: 起始文件
    :param end: 终止文件
    :param Period:周期
    :return: 具有pos值的点的集合
    '''
    point = get_Alldata_From_Native(start, end,Period)  # 读取模拟数据
    point = remove_Same_Point(point)
    samples=[]
    #计算每个路段两侧序号的平均值
    for e in edgeList:
        id1=0
        for i in verList[e.start].nearEdge:
            if edgeList[i].id!=e.id:
                id1+=edgeList[i].hilbertId
        id2=0
        for j in verList[e.end].nearEdge:
            if edgeList[j].id!=e.id:
                id2+=edgeList[j].hilbertId
        if len(verList[e.end].nearEdge)-1==0:
            t = e.start
            e.start = e.end
            e.end = t
        elif len(verList[e.start].nearEdge)-1!=0 and (id1/(len(verList[e.start].nearEdge)-1) > id2/(len(verList[e.end].nearEdge)-1)):
            t=e.start
            e.start=e.end
            e.end=t
    for p in point:
        temp=edgeList[p.edgeId].hilbertId+getDistance1(p, verList[edgeList[p.edgeId].start])/edgeList[p.edgeId].length
        rela_len=getDistance1(p, verList[edgeList[p.edgeId].start])/edgeList[p.edgeId].length
        if rela_len>1.0:
            print("匹配不合理")
        p.pos=float(format(temp,'.2f'))
        p1 = Sample(p.pos, p.time)
        p1.x=p.x
        p1.y=p.y
        p1.id=p.traId
        samples.append(p1)
        if p.pos>Point.max_pos:
            Point.max_pos=p.pos
        if p.pos<Point.min_pos:
            Point.min_pos=p.pos
    samples = list(set(samples))  # 去重
    return samples


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
        id = int(TraList[1])
        Traid = int(TraList[0])
        tempPoint = Point(x, y)
        tempPoint.traId = Traid
        tempPoint.time = time
        tempPoint.realTime = float(TraList[5])
        if time > Point.maxTime:
            Point.maxTime = time
        if time < Point.minTime:
            Point.minTime = time
        tempPoint.edgeId = id
        Point_list.append(tempPoint)
    return Point_list


def get_point(point_num, Period):
    '''
    :param point_num: 点的数量
    :param Period: 周期
    :return:返回数据点
    '''
    allPoint = []
    file = os.listdir("..//GeoData//P" + str(Period))
    for dir in file:
        file_name = "..//GeoData//P" + str(Period) + "//" + dir
        allPoint.extend(get_Data_From_Trajectory_real(file_name,Period))
    return allPoint[:point_num]


def give_Point_hilbert_real(edgeList, verList, point_num,Period=1):
    '''
    为每个点赋予相应的结合路段的希尔伯特值
    :param edgeList: 路段集合
    :param verList: 顶点集合
    :return: 具有希尔伯特值的点的集合
    '''
    point = get_point(point_num, Period)
    point = remove_Same_Point(point)
    samples=[]
    #计算每个路段两侧序号的平均值
    for e in edgeList:
        id1=0
        for i in verList[e.start].nearEdge:
            if edgeList[i].id!=e.id:
                id1+=edgeList[i].hilbertId
        id2=0
        for j in verList[e.end].nearEdge:
            if edgeList[j].id!=e.id:
                id2+=edgeList[j].hilbertId
        if len(verList[e.end].nearEdge)-1==0:
            t = e.start
            e.start = e.end
            e.end = t
        elif len(verList[e.start].nearEdge)-1!=0 and (id1/(len(verList[e.start].nearEdge)-1) > id2/(len(verList[e.end].nearEdge)-1)):
            t=e.start
            e.start=e.end
            e.end=t
    for p in point:
        temp=edgeList[p.edgeId].hilbertId+getDistance1(p, verList[edgeList[p.edgeId].start])/edgeList[p.edgeId].length
        rela_len=getDistance1(p, verList[edgeList[p.edgeId].start])/edgeList[p.edgeId].length
        if rela_len>1.0:
            print("匹配不合理")
        p.pos=float(format(temp,'.2f'))
        p1 = Sample(p.pos, p.time)
        p1.x=p.x
        p1.y=p.y
        p1.id=p.traId
        samples.append(p1)
        if p.pos>Point.max_pos:
            Point.max_pos=p.pos
        if p.pos<Point.min_pos:
            Point.min_pos=p.pos
    samples = list(set(samples))  # 去重
    return samples


def eq_instance(p1,p2):
    '''
    判断两个数据点的二维坐标（pos,t）是否相等
    :param p1:
    :param p2:
    :return: Bool
    '''
    if p1.pos == p2.pos and p1.time == p2.time:
        return True
    else:
        return False


def PModelTrain(point,maxError=3):
    '''
    训练rht
    :param point:数据点
    :param maxError:误差阈值
    :return: 模型，误差阈值
    '''
    X = []
    Y = []
    for p in point:
        temp = []
        temp.append(p.pos)
        temp.append(p.time)
        temp.append(p.hilbert)
        X.append(temp)
        Y.append(p.addr)
    eb = (DataBlock.BlockSize-1) * maxError
    CART.DataNum = len(X)  # 一期数据量
    boost1 = CART(stop_error=eb)
    print("开始训练")
    time11 = time.time()
    boost1.fit(np.array(X), np.array(Y))
    time21= time.time()
    print("rht构建模型时间：",time21 - time11)
    print('rht索引大小为', (boost1.leafNumber * 16 + (boost1.num_leafNode - boost1.leafNumber) * 20) / 1024, ' KB')
    return boost1,eb


def in_windows(row,col,x_scale,y_scale,q):
    '''
    判断网格是否在窗口内，若不在窗口内返回False
    :param row: 行
    :param col: 列
    :param x_scale: pos刻度
    :param y_scale: time刻度
    :param q: 窗口
    :return: Bool
    '''
    if row > 0 and col > 0:
        if x_scale[col] <= q[0] or x_scale[col - 1] >= q[1] or y_scale[row] <= q[2] or y_scale[row - 1] >= q[3]:
            return False
        else:
            return True
    elif col == 0 and row > 0:
        if x_scale[col] <= q[0] or y_scale[row] <= q[2] or y_scale[row - 1] >= q[3]:
            return False
        else:
            return True
    elif col > 0 and row == 0:
        if x_scale[col] <= q[0] or x_scale[col - 1] >= q[1] or y_scale[row] <= q[2]:
            return False
        else:
            return True
    else:
        if x_scale[col] <= q[0] or y_scale[row] <= q[2]:
            return False
        else:
            return True


def RangeQuery(q, model, eb,x_scale,y_scale):
    '''
    窗口查询
    :param q: 窗口,[min_pos,max_pos,min_time,max_time]
    :param model: 模型
    :param eb: 模型误差
    :param x_scale: pos维度的分割值
    :param y_scale: time维度的分割值
    :return:预测的地址
    '''
    ret=[]
    min_index=0
    max_index=len(x_scale)-1
    min_col=0
    min_row=0
    while min_index < max_index:
        mid = (min_index + max_index) // 2
        if q[0] >= x_scale[mid] and q[0] < x_scale[mid + 1]:
            min_col = mid + 1
            break
        elif q[0] < x_scale[mid]:
            max_index = mid
        elif q[0] > x_scale[mid]:
            min_index = mid
    min_index1 = 0
    max_index1 = len(y_scale) - 1
    while min_index1 < max_index1:
        mid1 = (min_index1 + max_index1) // 2
        if q[2] >= y_scale[mid1] and q[2] < y_scale[mid1 + 1]:
            min_row = mid1 + 1
            break
        elif q[2] < y_scale[mid1]:
            max_index1 = mid1
        elif q[2] > y_scale[mid1]:
            min_index1 = mid1
    begin={}  #起始值
    end={}  #终止值
    max_col=min_col
    max_row=min_row
    for scale in x_scale[min_col:]:
        col = x_scale.index(scale)
        max_col = col
        hil=hbg.hilbert_index(2,int(math.log(len(y_scale),2)),[min_row,col])  #希尔伯特值
        temp=hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0],temp[1],x_scale,y_scale,q)==False:  #判断该希尔伯特值是否为起点
            begin[hil] = [min_row, col]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [min_row, col]
        if scale >= q[1]:
            break
    for scale in y_scale[min_row+1:]:
        if q[3]<=y_scale[min_row]:
            break
        row=y_scale.index(scale)
        max_row =row
        hil = hbg.hilbert_index(2, int(math.log(len(y_scale), 2)), [row, min_col])  # 希尔伯特值
        temp = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0], temp[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为起点
            begin[hil] = [row, min_col]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [row, min_col]
        if scale >= q[3]:
            break
    rows=min_row+1
    while rows<=max_row:
        hil = hbg.hilbert_index(2, int(math.log(len(y_scale), 2)), [rows, max_col])
        temp = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0], temp[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为起点
            begin[hil] = [rows, max_col]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [rows, max_col]
        rows+=1
    cols=min_col+1
    while cols<max_col:
        hil = hbg.hilbert_index(2, int(math.log(len(y_scale), 2)), [max_row, cols])
        temp = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0], temp[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为起点
            begin[hil] = [max_row, cols]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [max_row, cols]
        cols+=1
    if len(begin)!=len(end):
        print("希尔伯特连续段非法!")
    else:
        order_begin=sorted(begin.items(),key=lambda x:x[0])  # 按希尔伯特值排序
        order_end=sorted(end.items(),key=lambda x:x[0])
        for i in range(len(order_begin)):
            tup=order_begin[i]
            tup1=order_end[i]
            cell=tup[1]  # 起始网格
            cell1=tup1[1]  # 终止网格
            pos=(x_scale[cell[1]]+Point.min_pos)/2
            time0=(y_scale[cell[0]]+Point.minTime)/2
            if cell[0]>0:
                time0=(y_scale[cell[0]]+y_scale[cell[0]-1])/2
            if cell[1]>0:
                pos=(x_scale[cell[1]]+x_scale[cell[1]-1])/2
            d=[]
            d.append([pos,time0,tup[0]])
            pre=int(model.predict(np.array(d)))-eb
            if pre<0:
                pre=0
            pos1 = (x_scale[cell1[1]] + Point.min_pos) / 2
            time1 = (y_scale[cell1[0]] + Point.minTime) / 2
            if cell1[0] > 0:
                time1 = (y_scale[cell1[0]] + y_scale[cell1[0] - 1]) / 2
            if cell1[1] > 0:
                pos1 = (x_scale[cell1[1]] + x_scale[cell1[1] - 1]) / 2
            d1=[]
            d1.append([pos1, time1, tup1[0]])
            pre1 = int(model.predict(np.array(d1))) + eb
            ret.append([pre,pre1])
    return ret


def initial_weight(hilbert_point, l):
    '''
    初始化样本权重
    :param hilbert_point: 字典,{希尔伯特值：数据点}
    :return: 样本权重
    '''
    d_order=sorted(hilbert_point)
    hn_min=d_order[0]
    hn_max=d_order[len(d_order)-1]
    fg=[]  # 前向间隙
    bg=[]  # 后向间隙
    for k in hilbert_point.keys():
        if d_order.index(k)!=0:
            inde=d_order.index(k)
            fg.append(k-d_order[inde-1])
        else:
            fg.append(hn_min)
    for k in hilbert_point.keys():
        if d_order.index(k) != len(d_order)-1:
            inde = d_order.index(k)
            bg.append(d_order[inde + 1]-k)
        else:
            bg.append(l*l-1-hn_max)

    w_sum=1/(hn_max-hn_min+l*l-1)
    sample_weight = w_sum/(np.array(fg) + np.array(bg))
    return sample_weight


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
    for i in range(DataBlock.BlockSize-1):
        mm = st.unpack('dd', tempStr[i * 16:i * 16 + 16])
        p=Sample(mm[0],mm[1])
        pointList.append(p)
    pointList.append(st.unpack('d',tempStr[8176:8184])[0])
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
            file = open(P.Dat,'r+b')
            file.seek(PointListId*8192)
            for i in range(DataBlock.BlockSize-1):
                p = PointList[i]
                file.write(st.pack('d',p.pos))
                file.write(st.pack('d', p.time))
            # file.write(st.pack('d', PointList[DataBlock.BlockSize-1]))
            file.close()
            temp = ReadBlockByIdForInsert(BlockId)
            buffer.InsertTest.append(temp)
            buffer.InsertId.append(BlockId)
            return temp


def bufferInit():
    '''
    初始化缓冲区
    :return:
    '''
    while len(buffer.InsertId)>0:
        PointList = buffer.InsertTest.pop(0)
        PointListId = buffer.InsertId.pop(0)
        file = open(P.Dat, 'r+b')
        file.seek(PointListId * 8192)
        for i in range(DataBlock.BlockSize-1):
            p = PointList[i]
            file.write(st.pack('d', p.pos))
            file.write(st.pack('d', p.time))
        file.close()


def range_filter(ret,q):
    '''
    过滤掉不属于该窗口的数据点
    :param ret: 预测的位置序列
    :param q:一个窗口,[min_pos,max_pos,min_time,max_time]
    :return: 查找到的在该窗口内的数据点
    '''
    res_block=set()
    res_point=[]  # 查询到的数据点
    # 获取块ID
    for i in range(len(ret)):
        start = mt.ceil(ret[i][0] * P.Expend)
        end = mt.ceil(ret[i][1] * P.Expend)
        start_id = start // (DataBlock.BlockSize - 1)  # 起始块号
        end_id = end // (DataBlock.BlockSize - 1)  # 终止块号
        if end_id > DataBlock.DataMaxBlock:
            end_id = DataBlock.DataMaxBlock
        for id in range(start_id, end_id + 1):
            res_block.add(id)
    # 判断是否使用溢出块
    if P.OutDataNum == 0:  # 没使用溢出块
        for block_id in res_block:
            point_list=IsBlockInBuffer(block_id)  # 获取该块的数据
            for i in range(DataBlock.BlockSize-1):
                p=point_list[i]
                if p.pos>=q[0] and p.pos<=q[1] and p.time>=q[2] and p.time<=q[3]:
                    res_point.append(p)
    else:
        for block_id in res_block:
            point_list=IsBlockInBuffer(block_id)  # 获取该块的数据
            for i in range(DataBlock.BlockSize-1):
                p=point_list[i]
                if p.pos>q[0] and p.pos<=q[1] and p.time>q[2] and p.time<=q[3]:
                    res_point.append(p)
            id=int(point_list[511])
            while id!=-1:
                point_list1 = IsBlockInBuffer(id)  # 获取该溢出块的数据
                for i in range(DataBlock.BlockSize - 1):
                    p = point_list1[i]
                    if p.pos >= q[0] and p.pos <= q[1] and p.time >= q[2] and p.time <= q[3]:
                        res_point.append(p)
                id=int(point_list1[511])
    return res_point


def range_filter_stack(ret,q):
    '''
    过滤掉不属于该窗口的数据点
    :param ret: 预测的位置序列
    :param q:一个窗口,[min_pos,max_pos,min_time,max_time]
    :return: 查找到的在该窗口内的数据点
    '''
    res_block=set()
    res_point=[]
    for i in range(len(ret)):
        start = ret[i][0]
        end = ret[i][1]
        start_id = start // (DataBlock.BlockSize - 1)  # 起始块号
        end_id = end // (DataBlock.BlockSize - 1)  # 终止块号
        if end_id > DataBlock.DataMaxBlock:
            end_id = DataBlock.DataMaxBlock
        for id in range(start_id, end_id + 1):
            res_block.add(id)
    # print("访问的磁盘块号：",res_block)
    for block_id in res_block:
        point_list=IsBlockInBuffer(block_id)  # 获取该块的数据
        for i in range(DataBlock.BlockSize-1):
            p=point_list[i]
            if p.pos>=q[0] and p.pos<=q[1] and p.time>=q[2] and p.time<=q[3]:
                res_point.append(p)
    return res_point


def PolynomialRegression(degree=2,**kwargs):
    '''
    多项式回归
    :param degree: 次数
    :param kwargs:
    :return:
    '''
    return make_pipeline(PolynomialFeatures(degree),linear_model.LinearRegression(**kwargs))


def write_data(file_name,points):
    '''
    将数据点的二维坐标写入指定文件中
    :param file_name:文件名
    :param points: 字典类型，键为希尔伯特值，值为数据点
    :return:
    '''
    d_order = sorted(points.items(), key=lambda x: x[0], reverse=False)  # 按字典集合中，每一个元组的键值元素排列
    file = open(file_name, 'wb')
    block_num = len(points) // (DataBlock.BlockSize-1) + 1  # 块数
    DataBlock.DataMaxBlock=block_num-1
    block_id=0
    while block_id<block_num:
        for i in range(block_id*(DataBlock.BlockSize-1),(block_id+1)*(DataBlock.BlockSize-1)):
            if i<len(points):
                p = d_order[i][1]
                b2 = st.pack('d', p.pos)
                b3 = st.pack('d', p.time)
                file.write(b2)
                file.write(b3)
            else:
                b2 = st.pack('d', 0)
                b3 = st.pack('d', 0)
                file.write(b2)
                file.write(b3)
        file.write(st.pack('d', -1))
        file.write(st.pack('d', -1))
        block_id+=1
    file.close()


def div(point,left,right,low,up):
    '''
    划分成四份
    :param point: 数据点
    :param left: 左边界
    :param right: 右边界
    :param low: 下边界
    :param up: 上边界
    :return:
    '''
    dr_low, dr_up, dl_low, dl_up = [], [], [], []
    pos_mid = (left + right) / 2
    time_mid = (low + up) / 2
    for i in range(len(point)):
        if point[i].pos <= pos_mid:
            if point[i].time <= time_mid:
                dl_low.append(point[i])  # 添加到左下角格子
            else:
                dl_up.append(point[i])  # 添加到左上角格子
        else:
            if point[i].time <= time_mid:
                dr_low.append(point[i])  # 添加到右下角格子
            else:
                dr_up.append(point[i])  # 添加到右上角格子
    return dr_low, dr_up, dl_low, dl_up


def curveOrderOne(point,left,right,low,up):
    '''
    等距离划分
    :param point: 数据点列表
    :param left: 左边界
    :param right: 右边界
    :param low: 下边界
    :param up: 上边界
    :return:网格阶数
    '''
    if len(point)<=1:
        return 0
    else:
        dr_low,dr_up,dl_low,dl_up=div(point,left,right,low,up)  # 对网格按中点划分成四个格子
        pos_mid = (left + right)/2
        time_mid = (low + up)/2
        d1=curveOrderOne(dr_low,pos_mid,right,low,time_mid)  # 对右下角格子递归
        d2=curveOrderOne(dr_up,pos_mid,right,time_mid,up)  # 对右上角格子递归
        d3=curveOrderOne(dl_low,left,pos_mid,low,time_mid)  # 对左下角格子递归
        d4=curveOrderOne(dl_up,left,pos_mid,time_mid,up)  # 对左上角格子递归
    return 1+max(d1,d2,d3,d4)


def grid_sort(point):
    '''
    对数据点划分网格，保证每个格子至多一个数据点
    :param point: 数据点
    :return: x_scale,y_scale
    '''
    X = []
    for p in point:
        temp = []
        temp.append(p.pos)
        temp.append(p.time)
        X.append(temp)
    features=np.array(X)
    feature_level = np.unique(features[:, 0])
    thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0
    x_scale = list(thresholds)
    x_scale.append(Point.max_pos)
    x_scale.sort()
    feature_level1 = np.unique(features[:, 1])
    thresholds1 = (feature_level1[:-1] + feature_level1[1:]) / 2.0
    y_scale = list(thresholds1)
    y_scale.append(Point.maxTime)
    y_scale.sort()
    return x_scale,y_scale


def hilbert_order_grid(x_scale,y_scale,point):
    '''
    对二维坐标排序
    :param x_scale: pos轴刻度
    :param y_scale: time轴刻度
    :param point: 数据点
    :return:
    '''
    sub = lambda a, b: a - b
    list1 = []
    list1.extend(x_scale)
    list1.pop(0)
    list2 = []
    list2.extend(x_scale)
    list2.pop()
    interval_x=[]
    min_x = x_scale[0] - Point.min_pos
    interval_x.append(min_x)
    interval_x.extend(list(map(sub, list1, list2)))
    list11 = []
    list11.extend(y_scale)
    list11.pop(0)
    list22 = []
    list22.extend(y_scale)
    list22.pop()
    interval_y=[]
    min_y = y_scale[0] - Point.minTime
    interval_y.append(min_y)
    interval_y.extend(list(map(sub, list11, list22)))
    if len(x_scale) >= len(y_scale):
        x1 = math.log(len(x_scale), 2)
        dx1 = int(x1)
        while x1 - dx1 != 0.0:
            max_in = np.argmax(np.mat(interval_x))  # 间隔最大
            scale_tmp=(x_scale[0]+Point.min_pos)/2
            if max_in>0:
                scale_tmp = (x_scale[max_in] + x_scale[max_in - 1]) / 2
            x_scale.append(scale_tmp)
            x_scale.sort()
            interval_x.insert(max_in,interval_x[max_in]/2)
            interval_x[max_in+1]=interval_x[max_in]/2
            x1 = math.log(len(x_scale), 2)
            dx1 = int(x1)
        while len(y_scale)<len(x_scale):
            max_in = np.argmax(np.mat(interval_y))  # 间隔最大
            scale_tmp = (y_scale[0] + Point.minTime) / 2
            if max_in > 0:
                scale_tmp = (y_scale[max_in] + y_scale[max_in - 1]) / 2
            y_scale.append(scale_tmp)
            y_scale.sort()
            interval_y.insert(max_in, interval_y[max_in] / 2)
            interval_y[max_in + 1] = interval_y[max_in] / 2
    else:
        x1=math.log(len(y_scale),2)
        dx1 = int(x1)
        while x1 - dx1 != 0.0:
            max_in = np.argmax(np.mat(interval_y))  # 间隔最大
            scale_tmp = (y_scale[0] + Point.minTime) / 2
            if max_in > 0:
                scale_tmp = (y_scale[max_in] + y_scale[max_in - 1]) / 2
            y_scale.append(scale_tmp)
            y_scale.sort()
            interval_y.insert(max_in, interval_y[max_in] / 2)
            interval_y[max_in + 1] = interval_y[max_in] / 2
            x1 = math.log(len(y_scale), 2)
            dx1 = int(x1)
        while len(x_scale)<len(y_scale):
            max_in = np.argmax(np.mat(interval_x))  # 间隔最大
            scale_tmp = (x_scale[0] + Point.min_pos) / 2
            if max_in > 0:
                scale_tmp = (x_scale[max_in] + x_scale[max_in - 1]) / 2
            x_scale.append(scale_tmp)
            x_scale.sort()
            interval_x.insert(max_in, interval_x[max_in] / 2)
            interval_x[max_in + 1] = interval_x[max_in] / 2
    hilbert_point={}
    for p in point:
        col=0
        row=0
        min_index=0
        max_index = len(x_scale)-1
        while min_index<max_index:
            mid = (min_index + max_index) // 2
            if  p.pos>x_scale[mid] and p.pos<=x_scale[mid+1]:
                col = mid+1
                break
            elif p.pos<=x_scale[mid]:
                max_index=mid
            elif p.pos>x_scale[mid]:
                min_index=mid
        min_index1 = 0
        max_index1 = len(y_scale) - 1
        while min_index1 < max_index1:
            mid1 = (min_index1 + max_index1) // 2
            if p.time > y_scale[mid1] and p.time <= y_scale[mid1 + 1]:
                row = mid1 + 1
                break
            elif p.time <= y_scale[mid1]:
                max_index1 = mid1
            elif p.time > y_scale[mid1]:
                min_index1 = mid1
        hilbert = hbg.hilbert_index(2, int(math.log(len(y_scale),2)), [row, col])
        if hilbert not in hilbert_point:
            hilbert_point[hilbert]=p
            p.hilbert=hilbert
        else:
            print("希尔伯特值重复！")
    print("二维坐标数据点数：",len(hilbert_point))
    d_order = sorted(hilbert_point.items(), key=lambda x: x[0], reverse=False)  # 按字典集合中，每一个元组的键值元素排列
    id = 0
    for tup in d_order:
        tup[1].addr = id
        id += 1
    return hilbert_point


def numPtn(pp,nm):
    '''
    求网格gc的dp
    :param pp: gc的父方向模式，二维数组
    :param nm:gc的序号
    :return:gc的dp，二维数组
    '''
    dp=[[pp[0][0],pp[0][1]],[pp[1][0],pp[1][1]]]
    if nm==1 or nm==2:
        return dp
    elif nm==0:
        i1,j1=0,0
        i3,j3=0,0
        for i in range(2):
            for j in range(2):
                if dp[i][j]==1:
                    i1, j1 =i,j
                if dp[i][j]==3:
                    i3, j3 =i,j
        dp[i1][j1],dp[i3][j3]=dp[i3][j3],dp[i1][j1]
        return dp
    elif nm==3:
        i0, j0 = 0, 0
        i2, j2 = 0, 0
        for i in range(2):
            for j in range(2):
                if dp[i][j] == 0:
                    i0, j0 = i, j
                if dp[i][j] == 2:
                    i2, j2 = i, j
        dp[i0][j0], dp[i2][j2] = dp[i2][j2], dp[i0][j0]
        return dp


def div_sort(point, left, right, low, up, idp, ssn):
    '''
    使用等距离划分网格对数据点排序
    :param point: 数据点列表
    :param left: 左边界
    :param right: 右边界
    :param low: 下边界
    :param up: 上边界
    :param idp: 初始方向模式
    :param ssn: 起始序号
    :return:网格阶数
    '''
    if len(point)<=1:
        if len(point)==1:
            point[0].addr=ssn
        return 0
    else:
        dr_low, dr_up, dl_low, dl_up = div(point, left, right, low, up)  # 对网格按中点划分成四个格子
        point_sort={}
        point_sort[idp[1][1]] = dr_low
        point_sort[idp[0][1]] = dr_up
        point_sort[idp[1][0]] = dl_low
        point_sort[idp[0][0]] = dl_up
        pos_mid = (left + right) / 2
        time_mid = (low + up) / 2
        orders=[]
        for k in range(4):
            if point_sort[k]==dr_low:
                idp1=numPtn(idp,k)
                order=div_sort(point_sort[k], pos_mid, right, low, time_mid, idp1, ssn)
                orders.append(order)
            elif point_sort[k]==dr_up:
                idp1=numPtn(idp,k)
                order=div_sort(point_sort[k], pos_mid, right, time_mid, up, idp1, ssn)
                orders.append(order)
            elif point_sort[k]==dl_low:
                idp1=numPtn(idp,k)
                order=div_sort(point_sort[k], left, pos_mid, low, time_mid, idp1, ssn)
                orders.append(order)
            else :
                idp1=numPtn(idp,k)
                order=div_sort(point_sort[k], left, pos_mid, time_mid, up, idp1, ssn)
                orders.append(order)
            ssn+=len(point_sort[k])
        return 1+max(orders)


def parentPattern(coor,coor_min,coor_max,ord,dp):
    '''
    求网格gc的父方向模式和序号
    :param coor: gc的行列号，[x,y]
    :param coor_min: gc所在网格的最小行列号，[xn,yn]
    :param coor_max: gc所在网格的最大行列号，[xm,ym]
    :param ord: 网格的阶数
    :param dp: 初始方向模式[[1,2],[0,3]]
    :return: gc的父方向模式和序号
    '''
    ppt=dp
    sn=0
    for i in range(1,ord+1):
        if i>1:
            ppt=numPtn(ppt,sn)
        x_cd,y_cd=1,1
        if coor[0]>=coor_min[0] and coor[0]<=(coor_min[0]+coor_max[0])//2:
            coor_max[0]=(coor_min[0]+coor_max[0])//2
        else:
            x_cd = 0
            coor_min[0]=(coor_min[0]+coor_max[0])//2+1
        if coor[1]>=coor_min[1] and coor[1]<=(coor_min[1]+coor_max[1])//2:
            y_cd=0
            coor_max[1]=(coor_min[1]+coor_max[1])//2
        else:
            coor_min[1]=(coor_min[1]+coor_max[1])//2+1
        sn=ppt[x_cd][y_cd]
    return ppt,sn


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


def read_datasets(path):
    '''
    读取数据集
    :param path: 数据存放的路径
    '''
    dirs = os.listdir(path)
    tempData = []  # 从文件中读取到的数据
    datasets=[]
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
        p=Sample(float(splitData[j][3]),float(splitData[j][4]))
        if p.pos>Point.max_pos:
            Point.max_pos=p.pos
        if p.pos<Point.min_pos:
            Point.min_pos=p.pos
        if p.time>Point.maxTime:
            Point.maxTime=p.time
        if p.time<Point.minTime:
            Point.minTime=p.time
        datasets.append(p)
    return datasets


def separate_train(Period=1):
    '''
    在模拟数据集上测试rht-sp
    :param Period: 周期
    '''
    path = "..//SIM//EP" + str(Period)
    point=read_datasets(path)
    x_scale,y_scale=grid_sort(point)
    hilbert_point=hilbert_order_grid(x_scale, y_scale, point)
    P.Dat = '..//DataSet//sim.dat'
    write_data(P.Dat,hilbert_point)  # 把数据写到外存
    boost1, eb = PModelTrain(point, maxError=2)  # 创建回归树模型
    range_query = read_range("..//SIM_DATA//range_sim.txt")  # 读取范围
    # 范围查询
    ans_num = 0
    for qr in range_query:
        ret = RangeQuery(qr, boost1, eb, x_scale, y_scale)  # 查询窗口的形式为(pos1,pos2),(t1,t2)
        res_point = range_filter(ret, qr)  # 获取查找到的在该窗口内的数据点，列表类型
        ans_num += len(res_point)
    print("找到的数据点数：", ans_num)
    real_num = 0
    for qr in range_query:
        for p in point:
            if p.pos >= qr[0] and p.pos <= qr[1] and p.time >= qr[2] and p.time <= qr[3]:
                real_num += 1
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：", buffer.InsertBlockNum)


def base_learner(point):
    '''
    base-learner层，训练几个异质的模型
    :param point: 数据点
    :return:模型列表
    '''
    print("开始训练base-learner")
    X = []  # 训练集的特征值
    Y = []  # 训练集的目标值
    for p in point:
        temp = []
        temp.append(p.pos)
        temp.append(p.time)
        temp.append(p.hilbert)
        X.append(temp)
        Y.append(p.addr)
    resTrain, eb = PModelTrain(point,maxError=1)
    train_x = []
    train_y=[]
    for p in point:
        temp = []
        temp.append(p.pos)
        temp.append(p.time)
        train_x.append(temp)
        train_y.append(p.addr)
    tree_model = tree.DecisionTreeRegressor(max_depth=7)  # cart树模型
    tree_model.fit(np.array(train_x), np.array(train_y))
    leaf_num = tree_model.get_n_leaves()  # 获取决策树的叶子节点数
    print("决策树叶子结点数", leaf_num)
    print("决策树结点数", tree_model.tree_.node_count)
    rr = linear_model.Ridge()  # 岭回归
    rr.fit(np.array(train_x), np.array(train_y))
    # 训练神经网络模型
    nn_model = tf.keras.Sequential()
    nn_model.add(tf.keras.layers.Dense(16, input_shape=(2,), activation='relu'))
    nn_model.add(tf.keras.layers.Dense(1))
    nn_model.compile(optimizer='adam', loss='mse')
    nn_model.fit(np.array(train_x), np.array(train_y), epochs=50)
    model_list = []  # 模型列表
    model_list.append(resTrain)
    model_list.append(tree_model)
    model_list.append(nn_model)
    model_list.append(rr)
    print("训练结束")
    return model_list


def rht_stack(model_list,Pe,point_num):
    '''
    在模拟数据上，rht-tk和单个模型的平均误差
    :param model_list: 单个模型列表
    :param Pe:周期
    :param point_num:第一期数据点个数
    '''
    path = "..//SIM//EP" + str(Pe)
    point = read_datasets(path)
    x_scale, y_scale = grid_sort(point)
    hilbert_point = hilbert_order_grid(x_scale, y_scale, point)  # 对数据点排序
    P.Expend=len(point)/point_num
    P.Dat = 'sim_stack'+str(Pe-1)+'.dat'
    DataBlock.DataMaxBlock = len(point) // (DataBlock.BlockSize - 1)
    write_data(P.Dat, hilbert_point)  # 把数据写到外存
    X=[]
    X1=[]
    for p in point:
        temp=[]
        temp.append(p.pos)
        temp.append(p.time)
        X1.append([p.pos,p.time])
        temp.append(p.hilbert)
        X.append(temp)
    pre_rht=model_list[0].predict(np.array(X))
    pre1=model_list[1].predict(np.array(X1))
    pre2 = model_list[2].predict(np.array(X1))
    pre3 = model_list[3].predict(np.array(X1))
    x_test=[]
    y_test=[]
    for i in range(len(X)):
        temp=[]
        temp.append(mt.ceil(pre_rht[i]*P.Expend))
        temp.append(mt.ceil(pre1[i]*P.Expend))
        temp.append(mt.ceil(pre2[i]*P.Expend))
        temp.append(mt.ceil(pre3[i]*P.Expend))
        x_test.append(temp)
        y_test.append(point[i].addr)
    t1 = time.time()
    model = PolynomialRegression(degree=4)  # 多项式模型
    model.fit(np.array(x_test), np.array(y_test))
    t2 = time.time()
    print("第二层构建时间：", t2 - t1)
    # 计算单个模型和stack的平均误差
    pre = model.predict(np.array(x_test))
    max_err = 0
    sum_err = 0
    sum_err1 = 0
    sum_err2 = 0
    sum_err3 = 0
    sum_err4 = 0
    for i in range(len(X)):
        err = abs(point[i].addr - mt.ceil(pre[i]))
        sum_err += err
        err1 = abs(point[i].addr - mt.ceil(pre_rht[i] * P.Expend))
        sum_err1 += err1
        err2 = abs(point[i].addr - mt.ceil(pre1[i] * P.Expend))
        sum_err2 += err2
        err3 = abs(point[i].addr - mt.ceil(pre2[i] * P.Expend))
        sum_err3 += err3
        err4 = abs(point[i].addr - mt.ceil(pre3[i] * P.Expend))
        sum_err4 += err4
        if err > max_err:
            max_err = err
    print("stack平均误差:", sum_err / len(X))
    print("rht平均误差:", sum_err1 / len(X))
    print("cart平均误差:", sum_err2 / len(X))
    print("神经网络平均误差:", sum_err3 / len(X))
    print("rr平均误差:", sum_err4 / len(X))
    eb = mt.ceil(sum_err / len(X))  # 平均误差
    return eb,point,x_scale,y_scale,model


def rht_stack_real(model_list,Pe,point_num):
    '''
    在真实数据上，rht-tk和单个模型的平均误差
    :param model_list: 单个模型列表
    :param Pe:周期
    :param point_num:第一期数据点个数
    '''
    path = "..//GEO//EP" + str(Pe)
    point = read_datasets(path)
    x_scale, y_scale = grid_sort(point)
    hilbert_point = hilbert_order_grid(x_scale, y_scale, point)  # 对数据点排序
    P.Expend = len(point) / point_num
    P.Dat = 'geo_stack.dat'
    DataBlock.DataMaxBlock = len(point) // (DataBlock.BlockSize - 1)
    write_data(P.Dat, hilbert_point)  # 把数据写到外存
    X=[]
    X1=[]
    for p in point:
        temp=[]
        temp.append(p.pos)
        temp.append(p.time)
        X1.append([p.pos,p.time])
        temp.append(p.hilbert)
        X.append(temp)
    pre_rht=model_list[0].predict(np.array(X))
    pre1=model_list[1].predict(np.array(X1))
    pre2 = model_list[2].predict(np.array(X1))
    pre3 = model_list[3].predict(np.array(X1))
    x_test=[]
    y_test=[]
    for i in range(len(X)):
        temp=[]
        temp.append(mt.ceil(pre_rht[i]*P.Expend))
        temp.append(mt.ceil(pre1[i]*P.Expend))
        temp.append(mt.ceil(pre2[i]*P.Expend))
        temp.append(mt.ceil(pre3[i]*P.Expend))
        x_test.append(temp)
        y_test.append(point[i].addr)
    t1 = time.time()
    model = PolynomialRegression(degree=4)  # 多项式模型
    model.fit(np.array(x_test), np.array(y_test))
    t2 = time.time()
    print("第二层构建时间：", t2 - t1)
    # 计算单个模型和stack的平均误差
    pre = model.predict(np.array(x_test))
    max_err = 0
    sum_err = 0
    sum_err1 = 0
    sum_err2 = 0
    sum_err3 = 0
    sum_err4 = 0
    for i in range(len(X)):
        err = abs(point[i].addr - mt.ceil(pre[i]))
        sum_err += err
        err1 = abs(point[i].addr - mt.ceil(pre_rht[i] * P.Expend))
        sum_err1 += err1
        err2 = abs(point[i].addr - mt.ceil(pre1[i] * P.Expend))
        sum_err2 += err2
        err3 = abs(point[i].addr - mt.ceil(pre2[i] * P.Expend))
        sum_err3 += err3
        err4 = abs(point[i].addr - mt.ceil(pre3[i] * P.Expend))
        sum_err4 += err4
        if err > max_err:
            max_err = err
    print("stack平均误差:", sum_err / len(X))
    print("rht平均误差:", sum_err1 / len(X))
    print("cart平均误差:", sum_err2 / len(X))
    print("神经网络平均误差:", sum_err3 / len(X))
    print("rr平均误差:", sum_err4 / len(X))
    eb = mt.ceil(sum_err / len(X))  # 平均误差
    return eb,point,x_scale,y_scale,model


def stack_predict(base_learn,meta_learn,d):
    x_test=np.array(d)[:,0:2]
    pre_rht = base_learn[0].predict(np.array(d))
    pre1 = base_learn[1].predict(np.array(x_test))
    pre2 = base_learn[2].predict(np.array(x_test))
    pre3 = base_learn[3].predict(np.array(x_test))
    X = []
    for i in range(len(d)):
        temp = []
        temp.append(mt.ceil(pre_rht[i]*P.Expend))
        temp.append(mt.ceil(pre1[i]*P.Expend))
        temp.append(mt.ceil(pre2[i]*P.Expend))
        temp.append(mt.ceil(pre3[i]*P.Expend))
        X.append(temp)
    pre=meta_learn.predict(np.array(X))
    return pre


def RangeQuery_stack(q, base_learn,meta_learn,eb,x_scale,y_scale):
    '''
    窗口查询
    :param q: 窗口,[min_pos,max_pos,min_time,max_time]
    :param base_learn: base_learner层
    :param meta_learn: meta_learner层
    :param eb: 模型误差
    :param x_scale: pos分割值
    :param y_scale: time分割值
    :return:预测的地址
    '''
    ret=[]
    min_index=0
    max_index=len(x_scale)-1
    min_col=0
    min_row=0
    # 找到窗口所在的最小行和最小列
    while min_index < max_index:
        mid = (min_index + max_index) // 2
        if q[0] >= x_scale[mid] and q[0] < x_scale[mid + 1]:
            min_col = mid + 1
            break
        elif q[0] < x_scale[mid]:
            max_index = mid
        elif q[0] > x_scale[mid]:
            min_index = mid
    min_index1 = 0
    max_index1 = len(y_scale) - 1
    while min_index1 < max_index1:
        mid1 = (min_index1 + max_index1) // 2
        if q[2] >= y_scale[mid1] and q[2] < y_scale[mid1 + 1]:
            min_row = mid1 + 1
            break
        elif q[2] < y_scale[mid1]:
            max_index1 = mid1
        elif q[2] > y_scale[mid1]:
            min_index1 = mid1
    begin={}  #起始值
    end={}  #终止值
    max_col=min_col
    max_row=min_row
    # 计算希尔伯特连续段
    for scale in x_scale[min_col:]:
        col = x_scale.index(scale)
        max_col = col
        hil=hbg.hilbert_index(2,int(math.log(len(y_scale),2)),[min_row,col])  #希尔伯特值
        temp=hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0],temp[1],x_scale,y_scale,q)==False:  #判断该希尔伯特值是否为起点
            begin[hil] = [min_row, col]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [min_row, col]
        if scale >= q[1]:
            break
    for scale in y_scale[min_row+1:]:
        if q[3]<=y_scale[min_row]:
            break
        row=y_scale.index(scale)
        max_row =row
        hil = hbg.hilbert_index(2, int(math.log(len(y_scale), 2)), [row, min_col])  # 希尔伯特值
        temp = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0], temp[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为起点
            begin[hil] = [row, min_col]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [row, min_col]
        if scale >= q[3]:
            break
    rows=min_row+1
    while rows<=max_row:
        hil = hbg.hilbert_index(2, int(math.log(len(y_scale), 2)), [rows, max_col])
        temp = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0], temp[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为起点
            begin[hil] = [rows, max_col]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [rows, max_col]
        rows+=1
    cols=min_col+1
    while cols<max_col:
        hil = hbg.hilbert_index(2, int(math.log(len(y_scale), 2)), [max_row, cols])
        temp = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil-1)
        if in_windows(temp[0], temp[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为起点
            begin[hil] = [max_row, cols]
        temp1 = hbg.hilbert_point(2,int(math.log(len(y_scale),2)),hil+1)
        if in_windows(temp1[0], temp1[1], x_scale, y_scale, q) == False:  # 判断该希尔伯特值是否为终点
            end[hil] = [max_row, cols]
        cols+=1
    # 预测连续段的端点所在位置
    if len(begin)!=len(end):
        print("希尔伯特连续段非法!")
    else:
        order_begin=sorted(begin.items(),key=lambda x:x[0])  # 按希尔伯特值排序
        order_end=sorted(end.items(),key=lambda x:x[0])
        for i in range(len(order_begin)):
            tup=order_begin[i]
            tup1=order_end[i]
            cell=tup[1]  # 起始网格
            cell1=tup1[1]  # 终止网格
            pos=(x_scale[cell[1]]+Point.min_pos)/2
            time0=(y_scale[cell[0]]+Point.minTime)/2
            if cell[0]>0:
                time0=(y_scale[cell[0]]+y_scale[cell[0]-1])/2
            if cell[1]>0:
                pos=(x_scale[cell[1]]+x_scale[cell[1]-1])/2
            d=[]
            d.append([pos,time0,tup[0]])
            pre=int(stack_predict(base_learn,meta_learn,d)[0])-eb
            if pre<0:
                pre=0
            pos1 = (x_scale[cell1[1]] + Point.min_pos) / 2
            time1 = (y_scale[cell1[0]] + Point.minTime) / 2
            if cell1[0] > 0:
                time1 = (y_scale[cell1[0]] + y_scale[cell1[0] - 1]) / 2
            if cell1[1] > 0:
                pos1 = (x_scale[cell1[1]] + x_scale[cell1[1] - 1]) / 2
            d1=[]
            d1.append([pos1, time1, tup1[0]])
            pre1 = int(stack_predict(base_learn,meta_learn,d1)[0]) + eb
            ret.append([pre,pre1])
    return ret


def stack_train(Period=1):
    '''
    在模拟数据上测试rht-tk
    :param Period: 周期
    '''
    path = "..//SIM//EP" + str(Period)
    point = read_datasets(path)
    print("数据点数：", len(point))
    x_scale, y_scale = grid_sort(point)
    hilbert_point = hilbert_order_grid(x_scale, y_scale, point)  # 对数据点排序
    P.Dat = '..//DataSet//sim_stack.dat'
    write_data(P.Dat, hilbert_point)  # 把数据写到外存
    time_b=time.time()
    model_list=base_learner(point)  # 训练几个异质的模型
    time_b1 = time.time()
    print("第一层构建时间：",time_b1-time_b)
    eb,point1,x_scale1,y_scale1,model=rht_stack(model_list,2,len(point))  # 对EP1以后的各个周期，训练集成模型
    range_query = read_range("..//SIM_DATA//range_sim.txt")  # 读取范围
    # 范围查询
    ans_num = 0
    for qr in range_query:
        ret = RangeQuery_stack(qr, model_list, model, eb, x_scale1,y_scale1)  # 查询窗口的形式为(pos1,pos2),(t1,t2)
        res_point = range_filter_stack(ret, qr)  # 获取查找到的在该窗口内的数据点，列表类型
        ans_num += len(res_point)
    print("找到的数据点数：", ans_num)
    real_num = 0
    for qr in range_query:
        for p in point1:
            if p.pos >= qr[0] and p.pos <= qr[1] and p.time >= qr[2] and p.time <= qr[3]:
                real_num += 1
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：", buffer.InsertBlockNum)


def separate_train_real(Period=1):
    '''
    在真实数据集上测试rht-sp
    :param PointNum: 轨迹点的数量
    :param Period: 周期
    :return:
    '''
    path = "..//GEO//EP" + str(Period)
    point = read_datasets(path)
    print("数据点数：",len(point))
    x_scale, y_scale = grid_sort(point)
    hilbert_point = hilbert_order_grid(x_scale, y_scale, point)  # 对数据点排序
    P.Dat = '..//DataSet//geo.dat'
    write_data(P.Dat,hilbert_point)  # 把数据写到外存
    boost1, eb = PModelTrain(point,maxError=3)  # 创建回归树模型
    range_query = read_range("..//GeoData//range_geo.txt")  # 读取范围
    # 范围查询
    ans_num = 0
    for qr in range_query:
        ret = RangeQuery(qr, boost1, eb, x_scale, y_scale)  # 查询窗口的形式为(pos1,pos2),(t1,t2)
        res_point = range_filter(ret, qr)  # 获取查找到的在该窗口内的数据点，列表类型
        ans_num += len(res_point)
    print("找到的数据点数：", ans_num)
    real_num = 0
    for qr in range_query:
        for p in point:
            if p.pos >= qr[0] and p.pos <= qr[1] and p.time >= qr[2] and p.time <= qr[3]:
                real_num += 1
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：", buffer.InsertBlockNum)


def stack_train_real(Period=1):
    '''
    在真实数据上测试rht-tk
    :param PointNum: 轨迹点的数量
    :param Period: 周期
    :return:
    '''
    path = "..//GEO//EP" + str(Period)
    point = read_datasets(path)
    print("数据点数：",len(point))
    point_num=len(point)
    x_scale, y_scale = grid_sort(point)
    hilbert_point = hilbert_order_grid(x_scale, y_scale, point)  # 对数据点排序
    P.Dat = '..//DataSet//geo_stack0.dat'
    write_data(P.Dat, hilbert_point)  # 把数据写到外存
    time_b=time.time()
    model_list=base_learner(point)  # 训练几个异质的模型
    time_b1 = time.time()
    print("第一层构建时间：",time_b1-time_b)
    eb,point1,x_scale1,y_scale1,model=rht_stack_real(model_list, 2, point_num)
    range_query = read_range("..//GeoData//range_geo.txt")  # 读取范围
    # 范围查询
    ans_num = 0
    for qr in range_query:
        ret = RangeQuery_stack(qr, model_list,model, eb, x_scale1,y_scale1)  # 查询窗口的形式为(pos1,pos2),(t1,t2)
        res_point = range_filter_stack(ret,qr)  # 获取查找到的在该窗口内的数据点，列表类型
        ans_num += len(res_point)
    print("找到的数据点数：", ans_num)
    real_num = 0
    for qr in range_query:
        for p in point1:
            if p.pos >= qr[0] and p.pos <= qr[1] and p.time >= qr[2] and p.time <= qr[3]:
                real_num += 1
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：", buffer.InsertBlockNum)


if __name__ == "__main__":
    # 模拟数据测试
    separate_train()
    stack_train()
    # 真实数据测试
    separate_train_real()
    stack_train_real()

