from numpy.random._examples.numba.extending import rg
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression, Ridge, LassoCV, RidgeCV, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC, SVR

import hilbertFromGithub as hbg
import HilbertForPoint as hbp
import math as mt
import random
import threading
import datetime
import sys
import numpy as np
import GTree as Gt
import NN1.NN
import time
import operator
import GTreeForEdge as gte
#import LearnModel.MyModelTree as mmt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import tensorflow as tf
import HilbertForPoint as hbe
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import datasets, svm


def trainTestSplit(trainingSet, trainingLabels, train_size):
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))#存放训练集的下标
    x_test = []     #存放测试集输入
    y_test= []      #存放测试集输出
    x_train = []    #存放训练集输入
    y_train = []    #存放训练集输出
    trainNum = int(totalNum * train_size) #划分训练集的样本数
    for i in range(trainNum):
        randomIndex = int(random.uniform(0,len(trainIndex)))
        x_test.append(trainingSet[trainIndex[randomIndex]])
        y_test.append(trainingLabels[trainIndex[randomIndex]])
        del(trainIndex[randomIndex])#删除已经放入测试集的下标
    for i in range(totalNum-trainNum):
        x_train.append(trainingSet[trainIndex[i]])
        y_train.append(trainingLabels[trainIndex[i]])
    return x_test,y_test,x_train,y_train

class Point:
    # 静态成员变量，用来存储当前路段中间的边界
    maxX = -1
    maxY = -1
    minX = 1000
    minY = 1000
    maxHilbert = 0
    minHilbert = float('inf')
    maxTime = 0
    minTime = float('inf')

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
        self.traId = None
        self.SortId = None
        self.edgeId = None
        self.nearEdge = set()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.x) * hash(self.y)

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


class Trajectory:
    def __init__(self, id, useId):
        self.PointList = []
        self.id = id
        self.useId = useId


# 将路段映射到一维空间上，这就是一维空间上路段
class hilbertEdge:
    def __init__(self, id, start, end, edgeId):
        self.id = id
        self.start = start
        self.end = end
        self.edgeId = edgeId

    def __str__(self):
        return "此时的希尔伯特路段为第" + str(self.id) + "个  起始位置和终止 位置分别为" + str(self.start) + " " + str(
            self.end) + "\n" + "对应的路段id为 " + str(self.edgeId)

    def __eq__(self, other):
        return self.id == other.id and self.start == other.start and self.end == other.end and self.edgeId == other.edgeId


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


def getDistance(lon1, lat1, lon2, lat2):
    '''
    求得两个点在空间上距离
    :param lon1:
    :param lat1:
    :param lon2:
    :param lat2:
    :return:
    '''
    # 将十进制转为弧度
    lon1, lat1, lon2, lat2 = map(mt.radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    aa = mt.sin(d_lat / 2) ** 2 + mt.cos(lat1) * mt.cos(lat2) * mt.sin(d_lon / 2) ** 2
    c = 2 * mt.asin(mt.sqrt(aa))
    r = 6371  # 地球半径，千米
    return c * r * 1000


# 计算两个相邻之间的路段经过希尔伯特排序之后的相差的距离的综合
def cal_Edge_Error1(edgelist, verlist):
    '''
    计算两个相邻路段经过排序之后的相差的值
    :param edgelist:
    :param verlist:
    :return:
    '''
    allCount = 0
    for edge in edgelist:
        temp_edge_list = []
        start = edge.start
        end = edge.end
        for near in verlist[start].nearEdge:
            if near not in temp_edge_list and near != edge.id:
                temp_edge_list.append(near)
        for near in verlist[end].nearEdge:
            if near not in temp_edge_list and near != edge.id:
                temp_edge_list.append(near)
        for near in temp_edge_list:
            # print(allCount)
            allCount += abs(edge.hilbert - edgelist[near].hilbert) - 1
    # print("此时路段顺序的总误差为 == " + str(allCount))
    return allCount


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
        x = float(TraList[2])
        y = float(TraList[3])
        time = int(TraList[0])
        id = int(TraList[1])
        tempPoint = Point(x, y)
        tempPoint.time = time
        if time > Point.maxTime:
            Point.maxTime = time
        if time < Point.minTime:
            Point.minTime = time
        tempPoint.traId = trajID
        tempPoint.edgeId = id
        Point_list.append(tempPoint)
        # print(tempPoint)
    return Point_list


def get_one_dimension_edge_hilbertEdge(sortEdge, edge_list):
    '''
    # 将经过排序后的路段转化为一维空间上间隔，同时将每个路段附上映射的一维空间上的位置，第一个输入是经过排序后的路段顺序，输出是路段的一维空间表示
    :param sortEdge:
    :param edge_list:
    :return:
    '''
    start = 0
    id = 0
    hil_edge_list = []
    for edge in sortEdge:
        end = start + edge[1].length
        temp_hil_edge = hilbertEdge(id, start, end, edge[1].id)
        hil_edge_list.append(temp_hil_edge)
        edge[1].hilbert = id
        start = end
        id += 1
    return hil_edge_list


# 从本地获取地图信息，输入是点的存放地址和路段的存放地址，输出为路段集合和点的集合
def get_Map_From_Native(PointStr, EdgeStr):
    file_of_point = open(PointStr)
    file_of_edge = open(EdgeStr)
    list_Ver = []
    list_Edge = []
    count = 0
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
        length = getDistance(ver_list[start].x, ver_list[start].y, ver_list[end].x, ver_list[end].y)
        cur_edge = Edge(id1, start, end, length)
        # if cur_edge in list_Edge:
        #     cur_edge.isTwo = 1
        #     count +=1
        # cur_edge.isTwo = int(tempList[4])
        #
        list_Edge.append(cur_edge)
        list_Ver[start].nearEdge.add(id1)
        list_Ver[end].nearEdge.add(id1)
    # print("重复的路段个数为"+str(count))
    return list_Ver, list_Edge


# 获取每个路段的中点，输入是点的集合和路段集合，输出是路段集合中路段的中点的集合。
def get_middle_point(list_edge, list_ver):
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
        tempPoint.edge_id = edge.id
        count = 0
        # if tempPoint in middle_point_list:
        #     edge.isTwo =1
        middle_point_list.append(tempPoint)
    return middle_point_list


def get_hilbert_N(pointlist, edge_list):
    '''
    为每个路段赋予一个独立的希尔伯特值
    :param pointlist:
    :param edge_list:
    :return:
    '''
    # 建立一个字典，每个希尔伯特是否对应单个值.
    n = 24
    # pointlist = get_Data_From_Trajectory(str1)
    mm = 1
    while (mm == 1):
        print("当前的阶数为" + str(n))
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
    # x相当于字典集合中遍历出来的一个元组。
    # print(type(d_order))
    # print(d_order)
    return n, d_order


# 将经过排序后的路段转化为一维空间上间隔，同时将每个路段附上映射的一维空间上的位置，第一个输入是经过排序后的路段顺序，输出是路段的一维空间表示
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


def get_Edge_n_Hop_Edge(edgeList, verList, n):
    '''
    为每个路段寻找他们的第几邻接路段
    :param edgeList: 路段集合
    :param verList: 点集合
    :param n 表示需要第几阶邻边
    :return:
    '''
    print("首先先添加第一邻边的点")
    # 首先为每个路段添加其邻边路段
    for edge in edgeList:
        edge.allNear.add(edge.id)
        # 为每个路段添加其邻边路段
        for id in verList[edge.start].nearEdge:
            if id != edge.id:
                edge.allNear.add(id)
                edge.nearEdge[0].add(edgeList[id])
        for id in verList[edge.end].nearEdge:
            if id != edge.id:
                edge.allNear.add(id)
                edge.nearEdge[0].add(edgeList[id])
    print("接下来添加1-n邻边的点")
    for i in range(1, n):
        for edge in edgeList:
            newSet = set()
            for nearedge in edge.nearEdge[i - 1]:
                for id in verList[nearedge.start].nearEdge:
                    if id not in edge.allNear:
                        edge.allNear.add(id)
                        newSet.add(edgeList[id])
                for id in verList[nearedge.end].nearEdge:
                    if id not in edge.allNear:
                        edge.allNear.add(id)
                        newSet.add(edgeList[id])
            edge.nearEdge.append(newSet)
    for edge in edgeList:
        edge.allNear = None
    print("每个路段的邻边添加完毕")


def get_Simulate_annealing_for_edge(edgeSort, edgeList):
    '''
    为经过希尔伯特排序后的点进行排序
    :param edgeSort:
    :param edgeList:
    :return:
    '''


def get_n_near_edge(n):
    '''
    获取计算了多少邻边的路段顺序
    :param n:表示多少阶的相邻路段
    :return:
    '''
    ver_list, edge_list = get_Map_From_Native("data//PPPoint.txt", "data//EEEdge.txt")
    get_Edge_n_Hop_Edge(edge_list, ver_list, n)
    middle_list = get_middle_point(edge_list, ver_list)
    n, edgeSort = get_hilbert_N(middle_list, edge_list)
    get_one_dimension_edge(edgeSort)
    cmpfun = operator.attrgetter("hilbert")
    edge_list.sort(key=cmpfun, reverse=False)
    return edge_list


def get_Alldata_From_Native(start, end):
    '''
    从本地获取所有的轨迹点，输出所有轨迹的集合
    :param start: 起始轨迹点的id
    :param end: 轨迹的坐标
    :return:
    '''
    allPoint = []
    for i in range(start, end):
        allPoint.extend(get_Data_From_Trajectory("../NN1//data//RealData//" + str(i) + ".txt", i))
    return allPoint


def remove_Same_Point(PointAll):
    '''
    对列表进行去重操作
    :param PointAll:
    :return:
    '''
    new_list = list(set(PointAll))
    return new_list


def give_Point_hilbert(edgeList ,verList,start,end):
    '''
    为每个点赋予相应的结合路段的希尔伯特值
    :param edgeList: 路段集合
    :param PointList: 轨迹点集合
    :param verList: 顶点集合
    :return: 具有希尔伯特值的点的集合
    '''
    point = get_Alldata_From_Native(start, end)
    point = remove_Same_Point(point)
    for epoint in point:
        edgeid = epoint.edgeId
        p1 = verList[edgeList[edgeid].start]
        p2 = verList[edgeList[edgeid].end]
        if p1 < p2:
            distance = getDistance(p1.x, p1.y, epoint.x, epoint.y)
            if distance > edgeList[edgeid].length:
                epoint.hilbert = edgeList[edgeid].hilbertId + 1
            else:
                epoint.hilbert = distance / edgeList[edgeid].length + edgeList[edgeid].hilbertId
        else:
            distance = getDistance(p2.x, p2.y, epoint.x, epoint.y)
            if distance > edgeList[edgeid].length:
                epoint.hilbert = edgeList[edgeid].hilbertId + 1
            else:
                epoint.hilbert = distance / edgeList[edgeid].length + edgeList[edgeid].hilbertId
        if epoint.hilbert < Point.minHilbert:
            Point.minHilbert = epoint.hilbert
        if epoint.hilbert > Point.maxHilbert:
            Point.maxHilbert = epoint.hilbert
    return point

def trainTestSplit(trainingSet, trainingLabels, train_size):
    '''
    按比例随机分割数据为测试集和训练集
    :param trainingSet:
    :param trainingLabels:
    :param train_size:
    :return:
    '''
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))#存放训练集的下标
    x_test = []     #存放测试集输入
    y_test= []      #存放测试集输出
    x_train = []    #存放训练集输入
    y_train = []    #存放训练集输出
    trainNum = int(totalNum * train_size) #划分训练集的样本数
    for i in range(trainNum):
        randomIndex = int(random.uniform(0,len(trainIndex)))
        x_test.append(trainingSet[trainIndex[randomIndex]])
        y_test.append(trainingLabels[trainIndex[randomIndex]])
        del(trainIndex[randomIndex])#删除已经放入测试集的下标
    for i in range(totalNum-trainNum):
        x_train.append(trainingSet[trainIndex[i]])
        y_train.append(trainingLabels[trainIndex[i]])
    return x_test,y_test,x_train,y_train

def build_model(): #建立两层神经网络
        input_dim = 1
        model = keras.Sequential([
            layers.Dense(32, input_shape=[input_dim,]),
            layers.Dense(32),
            layers.Dense(1)
        ])
        model.compile(loss='mse', metrics=['mae', 'mse'],
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model


def build_model2():  # 建立两层神经网络
    input_dim = 2
    model = keras.Sequential([
        layers.Dense(32, input_shape=[input_dim, ],activation='relu'),
        layers.Dense(32),
        layers.Dense(1)
    ])
    model.compile(loss='mse', metrics=['mae', 'mse'],
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

if __name__ == '__main__':
    ver_list, edge_list = get_Map_From_Native("data//PPPoint.txt", "data//EEEdge.txt")
    # get_Edge_n_Hop_Edge(edge_list, ver_list, n)
    middle_list = get_middle_point(edge_list, ver_list)
    n, edgeSort = get_hilbert_N(middle_list, edge_list)
    get_one_dimension_edge(edgeSort)
    point =  give_Point_hilbert(edge_list,ver_list,0,30)    #调整数据大小
    print("有多少个点",len(point))
    time1 = time.time()
    tree = gte.G_Tree_Create(50,50,1,2)
    for i in range(len(point)):
        gte.Insert_Point(tree,point[i])
    cmpfun = operator.attrgetter("blockName")
    point.sort(key=cmpfun, reverse=False)
    time2 = time.time()
    print("所需要花费的时间为",time2-time1)
    print("此时是后剪枝，设置偏差阈值的情况")
    for i in range(len(point)):
        if i == 0:
            point[i].blockId = 0
            continue
        else:
            if point[i].blockName == point[i - 1].blockName:
                point[i].blockId = point[i - 1].blockId
            else:
                point[i].blockId = point[i - 1].blockId + 1
    from sklearn.ensemble import *
    from sklearn.tree import DecisionTreeRegressor



    def DataSpilt():    #划分好数据集
        res = []
        for i in range(len(point)):
            temp = []
            temp.append(point[i].x)
            temp.append(point[i].y)
            temp.append(point[i].time - Point.minTime)
            # temp.append(PointList[i].time)
            temp.append(point[i].blockId)
            res.append(temp)
        return res


    # X = []
    # Y = []
    # for i in range(len(point)):
    #     temp = []
    #     temp.append(point[i].hilbert)
    #     temp.append(point[i].time)
    #     # temp.append(PointList[i].time)
    #     X.append(temp)
    #     Y.append(point[i].blockId)
    # Xtest1, Ytest1, X, Y = trainTestSplit(X, Y, 0.1)
    #



                #   训练    验证   测试
    def modelTrain(Train, Socre, Test, min_split=2, maxError=1000):
        # 异质回归器
        All_Model_list = []                                         #不同de模型
        #All_Model_list.append(DecisionTreeRegressor(max_depth=50))   #决策树
        #All_Model_list.append(LinearRegression())                     #线性回归器
        All_Model_list.append(SVR())                       #脊回归器

        #Train训练基础模型
        X = []
        Y = []
        for i in Train:
            temp = []
            temp.append(i[0])
            temp.append(i[1])
            temp.append(i[2])
            X.append(temp)
            Y.append(i[3])
        for i in range(len(All_Model_list)):
            All_Model_list[i].fit(np.array(X), np.array(Y))

        #Score训练集成模型
        X = []
        Y = []
        result = []
        for i in Socre:
            temp = []
            temp.append(i[0])
            temp.append(i[1])
            temp.append(i[2])
            X.append(temp)
            Y.append(i[3])
        for i in range(len(All_Model_list)):
            y_predict = All_Model_list[i].predict(X)
            result.append(y_predict)

        # #stacking 结果再进行两层神经网络
        model = build_model()
        model.fit(result, np.array(Y))

        print("接下来就是预测时间-----------------------")
        #测试集
        Xt = []
        Yt = []
        for i in Test:
            temp = []
            temp.append(i[0])
            temp.append(i[1])
            temp.append(i[2])
            Xt.append(temp)
            Yt.append(i[3])

        y_result = []
        for i in range(len(All_Model_list)):
            predict = All_Model_list[i].predict(Xt)
            y_result.append(np.array(predict))
        m_predict = model.predict(y_result)
        print(m_predict)
        # 误差
        error = 0
        error1 = 0
        Maxerror = 0
        Maxerror1 = 0
        count = 0
        # stacking模型下误差
        for i in range(len(Xt)):
            temperror = abs(m_predict[i] - Yt[i])
            error = temperror + error
            if temperror > Maxerror:
                Maxerror = temperror
            if temperror > 200:
                count += 1

        # 多层基础回归器嵌套后偏差和
        for i in range(len(All_Model_list)):
            for j in range(len(Xt)):
                temperror1 = abs(y_result[i][j] - Yt[j])
                error1 = temperror1 + error1
                if temperror1 > Maxerror1:
                    Maxerror1 = temperror1
        print("汪汪汪的次数为", count)
        print("此时的总误差为", error, error1)
        print("此时的最大误差为", Maxerror, Maxerror1)
        return error, error1, Maxerror, Maxerror1





    # 数据划分
    DataSet = DataSpilt()
    length = len(DataSet)
    print(length)
    DatasetNew = []
    DatasetNew.append(DataSet[0:int(length / 10 * 1)])
    DatasetNew.append(DataSet[int(length / 10 * 1):int(length / 10 * 2)])
    DatasetNew.append(DataSet[int(length / 10 * 2):int(length / 10 * 3)])
    DatasetNew.append(DataSet[int(length / 10 * 3):int(length / 10 * 4)])
    DatasetNew.append(DataSet[int(length / 10 * 4):int(length / 10 * 5)])
    DatasetNew.append(DataSet[int(length / 10 * 5):int(length / 10 * 6)])
    DatasetNew.append(DataSet[int(length / 10 * 6):int(length / 10 * 7)])
    DatasetNew.append(DataSet[int(length / 10 * 7):int(length / 10 * 8)])
    DatasetNew.append(DataSet[int(length / 10 * 8):int(length / 10 * 9)])
    DatasetNew.append(DataSet[int(length / 10 * 9):int(length + 1)])

    fileName = '10-811-m-e-1000.txt'
    fw = open(fileName, 'w')
    for i in range(6):
        Train = []
        Socre = []
        Test = []
        Train.extend(DatasetNew[(i) % 10])
        Train.extend(DatasetNew[(i + 1) % 10])
        Socre.extend(DatasetNew[(i + 2) % 10])
        Socre.extend(DatasetNew[(i + 3) % 10])
        Socre.extend(DatasetNew[(i + 4) % 10])
        Socre.extend(DatasetNew[(i + 5) % 10])
        Test.extend(DatasetNew[(i + 6) % 10])
        Test.extend(DatasetNew[(i + 7) % 10])
        Test.extend(DatasetNew[(i + 8) % 10])
        Test.extend(DatasetNew[(i + 9) % 10])
        e, e1, m, m1 = modelTrain(Train, Socre, Test)
        print(len(Train))
        print(len(Socre))
        print(len(Test))

        fw.write(str(e) + ',')
        fw.write(str(e1) + ',')
        fw.write(str(m) + ',')
        fw.write(str(m1) + '\n')




    # #adaboost
    # mm = AdaBoostRegressor(n_estimators=100,learning_rate=1,base_estimator=DecisionTreeRegressor(max_depth=100))
    # mm1 = DecisionTreeRegressor(max_depth=100)
    # mm.fit(X, Y)
    # mm1.fit(X, Y)


      #同质回归器
    # from sklearn.ensemble import RandomForestRegressor
    # listCartTree = []
    # result = []
    # for i in range(10):
    #     tempModel = DecisionTreeRegressor(max_depth=50)
    #     Xtest, Ytest, XTrain, YTrain = trainTestSplit(X, Y, 0.1)
    #     #temopmm = RandomForestRegressor(n_estimators=10)
    #     tempModel.fit(np.array(XTrain), np.array(YTrain))
    #     result.append(tempModel.predict(XTrain))
    #     listCartTree.append(tempModel)
    #
    # # #stacking 10个结果再进行两层神经网络
    # NewTrain = []
    # for j in range(len(XTrain)):
    #     temp = []
    #     for i in range(10):
    #         temp.append(result[i][j])
    #     NewTrain.append(np.array(temp))
    # print(np.array(YTrain))
    # model = build_model()
    # model.fit(result, np.array(YTrain))
    #
    # # 训练集和测试集不一样
    # y_result = []
    # for i in range(10):
    #     predict = listCartTree[i].predict(Xtest1)
    #     y_result.append(np.array(predict))
    # m_predict = model.predict(y_result)
    # # 误差
    # error = 0
    # error1 = 0
    # Maxerror = 0
    # Maxerror1 = 0
    # count = 0
    #
    # # stacking模型下误差
    # for i in range(len(Xtest1)):
    #     temperror = abs(m_predict[i] - Ytest1[i])
    #     error = temperror + error
    #     if temperror > Maxerror:
    #         Maxerror = temperror
    #     if temperror > 200:
    #         count += 1
    #
    # # 10层cart树的平均误差
    #
    # for i in range(10):
    #     for j in range(len(Xtest1)):
    #         temperror1 = abs(y_result[i][j] - Ytest1[j])
    #         error1 = temperror1 + error1
    #         if temperror1 > Maxerror1:
    #             Maxerror1 = temperror1
    # error1 = error1 / 10

    # # 异质回归器
    # All_Model_list = []                                         #不同de模型
    # All_Model_list.append(DecisionTreeRegressor(max_depth=50))   #决策树
    # All_Model_list.append(LinearRegression())                     #线性回归器
    # All_Model_list.append(Lasso())                                  #神经网络
    # All_Model_list.append(Ridge(normalize=True))                       #脊回归器
    # result = []
    # for i in range(len(All_Model_list)):
    #     Xtest, Ytest, XTrain, YTrain = trainTestSplit(X, Y, 0.1)
    #     All_Model_list[i].fit(XTrain, YTrain)
    #     y_predict = All_Model_list[i].predict(XTrain)
    #     result.append(y_predict)
    # # #stacking 结果再进行两层神经网络
    # NewTrain = []
    # for j in range(len(XTrain)):
    #     temp = []
    #     for i in range(len(All_Model_list)):
    #         temp.append(result[i][j])
    #     NewTrain.append(np.array(temp))
    #
    # model = build_model()
    # model.fit(result, np.array(YTrain))
    #
    #  #训练集和测试集不一样
    # y_result = []
    # for i in range(len(All_Model_list)):
    #     predict = All_Model_list[i].predict(Xtest1)
    #     y_result.append(np.array(predict))
    # m_predict = model.predict(y_result)
    # # 误差
    # error = 0
    # error1 = 0
    # Maxerror = 0
    # Maxerror1 = 0
    # count = 0
    #
    # # stacking模型下误差
    # for i in range(len(Xtest1)):
    #     temperror = abs(m_predict[i] - Ytest1[i])
    #     error = temperror + error
    #     if temperror > Maxerror:
    #         Maxerror = temperror
    #     if temperror > 200:
    #         count += 1
    #
    # # 异质回归器的平均误差
    #
    # for i in range(len(All_Model_list)):
    #     for j in range(len(Xtest1)):
    #         temperror1 = abs(y_result[i][j] - Ytest1[j])
    #         error1 = temperror1 + error1
    #         if temperror1 > Maxerror1:
    #             Maxerror1 = temperror1
    # error1 = error1 / 10



    # error = 0
    # error1 = 0
    # Maxerror = 0
    # Maxerror1 = 0
    # count = 0
    # result = mm.predict(Xtest1)
    # result1 = mm1.predict(Xtest1)
    # for i in range(len(Xtest1)):
    #
    #     temperror = abs(result[i] - Ytest1[i])
    #     error = temperror+error
    #     if temperror > Maxerror:
    #         Maxerror = temperror
    #     temperror1 = abs(result1[i] - Ytest1[i])
    #     error1 = temperror1 + error1
    #     if temperror1 > Maxerror1:
    #         Maxerror1= temperror1
    #     if temperror > 200:
    #         count += 1


    # print("汪汪汪的次数为", count)
    # print("此时的总误差为", error,error1)
    # print("此时的最大误差为", Maxerror,Maxerror1)
    #print("此时的叶节点为",mm.num_leafNode)
    # print("交叉验证",mm.cross_val_score(Xt,Yt,10))
    #print("预测的结果",mm.predict([X[25]]),"真正的结果",Y[25])
