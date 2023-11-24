import math as mt
import random
import threading
import datetime
import sys
import numpy as np


import time
import operator
# import DataSet.GTreeForPoint as gt
import sys  # 导入sys模块
import matplotlib.pyplot as plt

sys.setrecursionlimit(30000)
import hilbertFromGithub as hb


class Trajectory:
    def __init__(self, id, useId):
        self.PointList = []
        self.id = id
        self.useId = useId


# 点，可以是轨迹点，可以使交叉点
class Point:
    # 静态成员变量，用来存储当前轨迹的边界
    maxX = -1
    maxY = -1
    minX = 1000
    minY = 1000
    MaxHilbert = 0
    MinHilbert = float('inf')
    maxTime = 0
    minTime = float('inf')

    # 轨迹点的定义方式
    def __init__(self, x, y, time):
        '''

        :param x: 经度
        :param y: 纬度
        '''
        # x为经度，y为纬度,希尔伯特值和时间戳默认为-1
        self.x = x
        self.y = y
        self.time = time
        self.hilbert = None
        self.traId = None
        self.SortId = None

    def __str__(self):
        return str(self.x) + "  " + str(self.y) + "  " + str(self.time)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.x) * hash(self.y)

    def getMaxT(self):
        return self.maxTime

    def getMinT(self):
        return self.minTime

    def getMinH(self):
        return self.MinHilbert

    def getMaxH(self):
        return self.MaxHilbert

    def getMaxX(self):
        return self.maxX

    def getMinX(self):
        return self.minX

    def getMaxY(self):
        return self.maxY

    def getMinY(self):
        return self.minY


# 从本获取轨迹数据，输出为轨迹点的集合
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
        tempPoint = Point(x, y, time)
        if x > Point.maxX:
            Point.maxX = x
        if x < Point.minX:
            Point.minX = x
        if y > Point.maxY:
            Point.maxY = y
        if y < Point.minY:
            Point.minY = y
        if time > Point.maxTime:
            Point.maxTime = time
        if time < Point.minTime:
            Point.minTime = time
        tempPoint.traId = trajID
        Point_list.append(tempPoint)
        # print(tempPoint)
    return Point_list


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


def sort_Point_By_One_Attribute(PointAll, atr):
    '''
    对列表按照某一特定的属性进行排序
    :param PointAll: 所有点的集合
    :param atr: 需要进行排序的点的集合
    :return:
    '''
    cmpfun = operator.attrgetter(atr)
    PointAll.sort(key=cmpfun, reverse=False)


def remove_Same_Point(PointAll):
    '''
    对列表进行去重操作
    :param PointAll:
    :return:
    '''
    new_list = list(set(PointAll))
    return new_list


def cal_min_X_distance(PointAll):
    '''
    求所有x之间的最小距离，不包括0
    :param PointAll:
    :return:
    '''
    minDistance = 1000
    sort_Point_By_One_Attribute(PointAll, "x")
    for i in range(len(PointAll) - 1):
        distance = abs(PointAll[i].x - PointAll[i + 1].x)
        if distance == 0:
            continue
        else:
            if distance < minDistance:
                minDistance = distance
    print("最小距离为", minDistance)
    print("最大距离为", (PointAll[0].getMaxX() - PointAll[0].getMinX()))

    return np.ceil((PointAll[0].getMaxX() - PointAll[0].getMinX()) / minDistance)


def cal_min_Y_distance(PointAll):
    '''
    求所有y之间的最小距离，不包括0
    :param PointAll:
    :return:
    '''
    minDistance = 1000
    sort_Point_By_One_Attribute(PointAll, "y")
    for i in range(len(PointAll) - 1):
        distance = abs(PointAll[i].y - PointAll[i + 1].y)
        if distance == 0:
            continue
        else:
            if distance < minDistance:
                minDistance = distance
    print("最小距离为", minDistance)
    print("最大距离为", (PointAll[0].getMaxY() - PointAll[0].getMinY()))

    return np.ceil(abs(PointAll[0].getMaxY() - PointAll[0].getMinY()) / minDistance)


def get_Hilbert_For_Point_3D(PointAll, n):
    '''
    为每一个点赋予他们的希尔伯特值,三维
    :param PointAll:所有的点的集合
    :param n: 希尔伯特的阶数
    :return: 赋予希尔伯特值的点
    '''
    count = 0
    newSet = {}
    for point in PointAll:
        Xinter = (point.maxX - point.minX) / pow(2, n)
        Xnum = int((point.x - point.minX) / Xinter)

        Yinter = (point.maxY - point.minY) / pow(2, n)
        Ynum = int((point.y - point.minY) / Yinter)

        Tinter = (point.maxTime - point.minY) / pow(2, n)
        Tnum = int((point.time - point.minTime) / Tinter)
        # 三维空间下的希尔伯特值
        hilbert = hb.hilbert_index(3, n, [Xnum, Ynum, Tnum])
        point.hilbert = hilbert
        if hilbert in newSet:
            count += 1
            print("此时的点为", point, "字典里面点为", newSet[hilbert])
            print("前一个点的在希尔伯特空间的位置为", Xnum, Ynum)
            print("字典中的点在希尔伯特空间的位置为", np.ceil((newSet[hilbert].x - point.minX) / Xinter),
                  np.ceil((newSet[hilbert].y - point.minY) / Yinter))
            # print("此时的希尔伯特值为",hilbert)
        else:
            newSet[hilbert] = point

    print("重复的次数为", count)


def get_Hilbert_For_Point_2D(PointAll, n):
    '''
    为每一个点赋予他们的希尔伯特值，二维
    :param PointAll:所有的点的集合
    :param n: 希尔伯特的阶数
    :return: 赋予希尔伯特值的点
    '''
    count = 0
    newSet = {}
    for point in PointAll:
        Xinter = (point.maxX - point.minX) / pow(2, n)
        Xnum = int((point.x - point.minX) / Xinter)

        Yinter = (point.maxY - point.minY) / pow(2, n)
        Ynum = int((point.y - point.minY) / Yinter)
        # 二维空间下的希尔伯特值
        hilbert = hb.hilbert_index(2, n, [Xnum, Ynum])
        point.hilbert = hilbert
        if hilbert in newSet:
            count += 1
            print("此时的点为", point, "字典里面点为", newSet[hilbert])
            print("前一个点的在希尔伯特空间的位置为", Xnum, Ynum)
            print("字典中的点在希尔伯特空间的位置为", np.ceil((newSet[hilbert].x - point.minX) / Xinter),
                  np.ceil((newSet[hilbert].y - point.minY) / Yinter))

        else:
            newSet[hilbert] = point

    print("重复的次数为", count)


def HilBert(n, x, y):
    '''
    递归获取每个点的希尔伯特值
    :param n: 当前希尔伯特的阶数
    :param x: 此时的x值
    :param y: 此时的y值
    :return:
    '''
    if n == 0:
        return 1
    m = 1 << (n - 1)
    # 此时在左下角
    if x <= m and y <= m:
        return HilBert(n - 1, x, y)
    # 此时在右下角
    if x > m >= y:
        return 3 * m * m + HilBert(n - 1, m - y + 1, m * 2 - x + 1)
    # 此时在左上角
    if x <= m < y:
        return 1 * m * m + HilBert(n - 1, x, y - m)
    # 此时在右上角
    if x > m and y > m:
        return 2 * m * m + HilBert(n - 1, x - n, y - m)


def point_to_hilbert(x, y, order=16):
    hilbert_map = {
        'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},
        'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},
        'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},
        'd': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},
    }
    current_square = 'a'
    position = 0
    for i in range(order - 1, -1, -1):
        position <<= 2
        quad_x = 1 if x & (1 << i) else 0
        quad_y = 1 if y & (1 << i) else 0
        quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
        position |= quad_position
    return position


def hilbert_to_point(d, order=16):
    '''
    将希尔伯特转换成点的坐标，不过估计用不到
    :param d: 希尔伯特值
    :param order:
    :return:
    '''
    un_hilbert_map = {
        'a': {0: (0, 0, 'd'), 1: (0, 1, 'a'), 3: (1, 0, 'b'), 2: (1, 1, 'a')},
        'b': {2: (0, 0, 'b'), 1: (0, 1, 'b'), 3: (1, 0, 'a'), 0: (1, 1, 'c')},
        'c': {2: (0, 0, 'c'), 3: (0, 1, 'd'), 1: (1, 0, 'c'), 0: (1, 1, 'b')},
        'd': {0: (0, 0, 'a'), 3: (0, 1, 'c'), 1: (1, 0, 'd'), 2: (1, 1, 'd')}
    }
    current_square = 'a'
    x = y = 0
    for i in range(order - 1, -1, -1):
        # 3的二进制为11，然后左移2i倍，与d取按位与后右移2i倍，得到象限编码
        mask = 3 << (2 * i)
        quad_position = (d & mask) >> (2 * i)

        quad_x, quad_y, current_square = un_hilbert_map[current_square][quad_position]
        print(quad_x, quad_y)

        # 不断累加x，y的值，最后总得到解码结果
        x |= 1 << i if quad_x else 0
        y |= 1 << i if quad_y else 0

    return x, y


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


def get_DataSet_2D_Hilbert(start, end):
    '''
    获取数据集合，以二维希尔伯特值的方式获取
    :param start: 起始轨迹段
    :param end: 结束轨迹段
    :return:
    '''
    point = get_Alldata_From_Native(start, end)
    point = remove_Same_Point(point)
    get_Hilbert_For_Point_2D(point, 27)
    sort_Point_By_One_Attribute(point, "hilbert")
    return point


def get_DataSet_3D_Hilbert(start, end):
    '''
    获取数据集合，以三维希尔伯特值的方式获取
    :param start: 起始轨迹段
    :param end: 结束轨迹段
    :return:
    '''
    point = get_Alldata_From_Native(start, end)
    point = remove_Same_Point(point)
    print("接下来是为每个点寻找k个近邻点")
    get_KPoint_For_Every_Point(point, 4, 400)
    get_Hilbert_For_Point_3D(point, 27)
    sort_Point_By_One_Attribute(point, "hilbert")
    return point


def get_DataSet_Gtree(start, end, Num, dim):
    '''
    获取数据集合，以二维的G树空间,每个节点的固定有最大容量
    :param start: 起始轨迹段
    :param end: 结束轨迹段
    :param dim:此时的G树的纬度
    :return:
    '''
    point = get_Alldata_From_Native(start, end)
    point = remove_Same_Point(point)
    print("接下来是为每个点寻找k个近邻点")
    get_KPoint_For_Every_Point(point, 4, 400)
    tree = gt.G_Tree_Create(100, 100, Num, dim)
    for i in range(len(point)):
        gt.Insert_Point(tree, point[i])
    sort_Point_By_One_Attribute(point, "blockName")
    BlockId = 0
    for i in range(len(point)):
        if i == 0:
            point[i].blockId = 0;
            continue
        else:
            if point[i].blockName == point[i - 1].blockName:
                point[i].blockId = point[i - 1].blockId
            else:
                point[i].blockId = point[i - 1].blockId + 1
    return point


def get_KPoint_For_Every_Point(PointList, k,inter):
    '''
    给每个点寻找最近的k个点
    :param PointList:
    :param k :表示需要选择相邻的k个点
    :param inter:表示分为多少间隔
    :return:
    '''
    Maxinter = inter
    Xinter = (Point.maxX - Point.minX) / Maxinter
    Yinter = (Point.maxY - Point.minY) / Maxinter
    Tinter = (Point.maxTime - Point.minTime) / Maxinter
    Grid = {}
    for point in PointList:
        Xnum = int((point.x - Point.minX) // Xinter)
        Ynum = int((point.y - Point.minY) // Yinter)
        Tnum = int((point.time - Point.minTime) // Tinter)
        temp = (Xnum, Ynum, Tnum)
        if temp in Grid.keys():
            Grid[temp].append(point)
        else:
            Grid[temp] = []
            Grid[temp].append(point)
    maxNum = k
    for point in PointList:
        Isvisit = np.zeros((Maxinter, Maxinter, Maxinter))
        Xnum = int((point.x - Point.minX) // Xinter)
        Ynum = int((point.y - Point.minY) // Yinter)
        Tnum = int((point.time - Point.minTime) // Tinter)
        near = {}
        BFS(point, Xnum,Ynum,Tnum,Grid,Isvisit,near,maxNum,Maxinter,0,MaxDepth=5)
        near = sorted(near.items(),key=lambda x:x[0])
        nearPoint =[]
        for i in range(len(near)):
            if i==k:
                break;
            else:
                nearPoint.append(near[i][1])
        point.nearPoint = nearPoint


def BFS(point, Xnum, Ynum, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep,MaxDepth =5):
    '''
    递归获取当前点的邻接点，同时规定搜索边界在一定值范围，否则会导致递归深度过大
    :param point:当前的点
    :param Xnum:当前点所在X方向的位置
    :param Ynum:当前点所在Y方向的位置
    :param Tnum:当前点所在T方向的位置
    :param Grid:网格索引
    :param isvisit:访问数组
    :param near:邻近点
    :param MaxNum:最大数目
    :param Maxinter:最大的间隔数
    :param MaxDepth:最大的递归深度，在代码的第一行，目前是5
    :return:
    '''
    if (Xnum, Ynum, Tnum) not in Grid.keys() or nowdep ==MaxDepth:
        return
    if Xnum < 0 or Xnum >= Maxinter or Ynum < 0 or Ynum >= Maxinter or Tnum < 0 or Tnum >= Maxinter or \
            isvisit[Xnum][Ynum][Tnum] == 1 :
        return
    isvisit[Xnum][Ynum][Tnum] = 1
    temp = (Xnum, Ynum, Tnum)

    list1 = Grid[temp]
    for tempPoint in Grid[temp]:
        if tempPoint==point:
            continue
        if tempPoint not in near:
            near[mt.sqrt(getDistance(point.x, point.y, tempPoint.x, tempPoint.y) ** 2 + (
                        point.time - tempPoint.time) ** 2)] = tempPoint
    if (len(near) < MaxNum):
        nowdep+=1
        BFS(point, Xnum + 1, Ynum, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        # 和当前点属于同一平面time的点
        BFS(point, Xnum + 1, Ynum + 1, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum + 1, Ynum - 1, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum + 1, Ynum, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum - 1, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum + 1, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum - 1, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum + 1, Tnum, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        # 属于上一个平面的
        BFS(point, Xnum + 1, Ynum + 1, Tnum + 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum + 1, Ynum - 1, Tnum + 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum + 1, Ynum, Tnum + 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum, Tnum+1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum - 1, Tnum+1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum + 1, Tnum+1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum - 1, Tnum+1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum + 1, Tnum+1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum, Tnum + 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        #属于下一个平面的
        BFS(point, Xnum + 1, Ynum + 1, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum + 1, Ynum - 1, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum + 1, Ynum, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum - 1, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum - 1, Ynum + 1, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum - 1, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum + 1, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
        BFS(point, Xnum, Ynum, Tnum - 1, Grid, isvisit, near, MaxNum, Maxinter,nowdep)
#test
def test():
    time1 = time.time()
    point = get_Alldata_From_Native(0, 535)
    point = remove_Same_Point(point)
    print("接下来是为每个点寻找k个近邻点")
    get_KPoint_For_Every_Point(point,4,400)
    print(point[1])
    time2 = time.time()
    print("花费的时间为",time2-time1)
    return point

def plot_point(point):
    '''
    给每个point画图
    :param point:
    :return:
    '''
    X = []
    Y = []
    Z = []
    coot = 0
    for i in point:
        X.append(i.x)
        Y.append(i.y)
        Z.append(coot)
        coot += 1
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X,Y,Z)
    plt.show()
if __name__ == '__main__':
    # test()
    index  = 2000
    count =0
    while index>1:
        count+=1
        index *=0.999
    print(count)
    # point = get_DataSet_Gtree(0, 535, 2, 2)
    # point = get_DataSet_2D_Hilbert(0, 535)
    # 为每个点找一个邻接点
    print("此时的大小为",np.exp(-(20000)/200))

