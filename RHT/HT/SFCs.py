# standard libraries
import getopt
import os
import struct
import time

import numpy as np
import sys
# private libraries
import Rtreefromgithub
import hilbertFromGithub as hb
import scanRange

global root
global Bvalue



class DataBlock:
    DataMaxBlock = 0
    OutDataBlockSize = 0
    BlockSize = 341

def IsBlockInBuffer(BlockId):
    if BlockId in buffer.InsertTest:
        return
    else:
        if len(buffer.InsertTest) < 128:
            buffer.InsertBlockNum += 1
            buffer.InsertTest.append(BlockId)
        else:
            buffer.InsertBlockNum += 2
            buffer.InsertTest.pop(0)
            buffer.InsertTest.append(BlockId)

class buffer:
    IndexBuffer = []
    DataIdBuffer = []
    DataBuffer = {}
    InsertTest = []
    InsertBlockNum = 0

def getDataByBlockId(BlockId, file, blockNum):
    '''
    根据块号从本地数据，读取数据，如果缓冲区有数据，那么就直接在缓冲区读，如果没有，就从本地读取
    :param BlockId: 当前需要读取的数据块号
    :param file: 当前文件流
    :param blockNum:当前读取多了多少块号
    :return:
    '''
    if BlockId in buffer.DataBuffer.keys():
        ssm = buffer.DataBuffer[BlockId]
        return ssm, blockNum
    else:
        file.seek(BlockId * 8192, 0)
        ssm = file.read(8192)
        # 如果当前缓冲区未满，直接添加
        if buffer.DataIdBuffer.__sizeof__() < 128:
            buffer.DataIdBuffer.append(BlockId)
            buffer.DataBuffer[BlockId] = ssm
        else:
            # 如果缓冲区满了，那么就删除第一个元素，同时将此次读入数据添加
            buffer.DataBuffer.pop(buffer.DataIdBuffer[0])
            buffer.DataIdBuffer.pop(0)
            buffer.DataIdBuffer.append(BlockId)
            buffer.DataBuffer[BlockId] = ssm
        return ssm, blockNum + 1

def IsdataInRange(ssm, range1):
    '''
    判断该数据块是否有范围内的点
    :param ssm: 数据块
    :param range1: 范围
    :return: 范围内的点的数量
    '''
    Num = 0
    for i in range(DataBlock.BlockSize):
        p = struct.unpack('ddd', ssm[i * 24:i * 24 + 24])
        if p[0] != 0 and p[1] != 0 and p[2] != 0 and range1.x1 <= p[0] <= range1.x2 and range1.y1 <= p[
            1] <= range1.y2 and range1.time1 <= p[
            2] <= range1.time2:
            Num+=1
    return Num

def IfDataBlockHasData(ssm, p):
    '''
    数据块中是否包含数据
    :param ssm: 数据块
    :param p: 点p
    :return:
    '''
    for i in range(DataBlock.BlockSize):
        ssmm = struct.unpack('ddd', ssm[i * 24:i * 24 + 24])
        if ssmm[0] == p.x and ssmm[1] == p.y and ssmm[2] == p.time:
            return False
    return True




class Trajectory:
    def __init__(self, id, useId):
        self.PointList = []
        self.id = id
        self.useId = useId


# 点，可以是轨迹点，可以使交叉点
class Point:
    # 静态成员变量，用来存储当前轨迹的边界
    maxX = 116.7285615
    maxY = 40.1799936
    minX = 116.0800006
    minY = 39.6800104000001
    MaxHilbert = 0
    MinHilbert = float('inf')
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
        return self.x == other.x and self.y == other.y and self.time == other.time

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

    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.time)


def get_Data_From_Trajectory(str1, initime=1,Period = 1):
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
        time = float(TraList[5])-timeMap[initime]
        id = int(TraList[1])
        Traid = int(TraList[0])
        tempPoint = Point(x, y)
        tempPoint.traId = Traid
        tempPoint.time = time
        if time > Point.maxTime:
            Point.maxTime = time
        if time < Point.minTime:
            Point.minTime = time
        tempPoint.traId = 1
        tempPoint.edgeId = id

        Point_list.append(tempPoint)
        # print(tempPoint)
    return Point_list




def get_NextP_Point(start, end, m,maxlength=10000000):
    allPoint = []
    file = os.listdir("../GeoDate//P"+ str(m))
    end = len(file)
    for dir in file:
        allPoint.extend(get_Data_From_Trajectory("../GeoDate//P" + str(m) + "//" +dir, m))
    # for i in range(start, end):
    #     # allPoint.extend(get_Data_From_Trajectory("../NN1//data//RealData//" + str(i) + ".txt", i))
    #     allPoint.extend(get_Data_From_Trajectory("Real_Data//" + str(m) + "//" + str(i) + ".txt", i))
    allPoint = remove_Same_Point(allPoint)
    print('本周期的点数为',len(allPoint))

    return allPoint[0:maxlength]

def get_sim_Point(start, end, m,maxlength=10000000):
    allPoint = []
    file = os.listdir("../SIM_DATA//P"+ str(m))
    end = len(file)
    for dir in file:
        allPoint.extend(get_Data_From_Trajectory("../SIM_DATA//P" + str(m) + "//" +dir, m))
    # for i in range(start, end):
    #     # allPoint.extend(get_Data_From_Trajectory("../NN1//data//RealData//" + str(i) + ".txt", i))
    #     allPoint.extend(get_Data_From_Trajectory("Real_Data//" + str(m) + "//" + str(i) + ".txt", i))
    allPoint = remove_Same_Point(allPoint)
    print('本周期的点数为',len(allPoint))

    return allPoint[0:maxlength]



def remove_Same_Point(PointAll):
    '''
    对列表进行去重操作
    :param PointAll:
    :return:
    '''
    new_list = list(set(PointAll))
    return new_list



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
            #print("此时的点为", point, "字典里面点为", newSet[hilbert])
            #print("前一个点的在希尔伯特空间的位置为", Xnum, Ynum)
            #print("字典中的点在希尔伯特空间的位置为", np.ceil((newSet[hilbert].x - point.minX) / Xinter),
            #      np.ceil((newSet[hilbert].y - point.minY) / Yinter))

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


#构建r树
global root
global Bvalue
#节点溢出
def handleOverFlow(node):
    global root
    global Bvalue

    # split node into two new nodes
    nodes = node.split()
    # if root node is overflow, new root need to build
    if node.paren == None:
        root = Rtreefromgithub.Branch(Bvalue, node.level + 1, nodes[0])
        root.addChild(nodes[0])
        root.addChild(nodes[1])
        root.childList[0].paren = root
        root.childList[1].paren = root
    else:
        # update the parent node
        parent = node.paren
        parent.childList.remove(node)
        parent.childList += nodes
        # check whether parent node is overflow
        if parent.isOverFlow():
            handleOverFlow(parent)


# insert a point to a node
def insert(node, point):
    # if the node is a leaf, add this point
    if isinstance(node, Rtreefromgithub.Leaf):
        node.addChild(point)
        if node.isOverFlow():
            handleOverFlow(node)

    # if the node is a branch, choose a child to add this point
    elif isinstance(node, Rtreefromgithub.Branch):
        node.update(point)
        childNode = node.chooseChild(point)
        insert(childNode, point)

    else:
        pass


def buildRtree(allpoint, *B):
    global root
    global Bvalue

    Bvalue = 341

    if len(B) == 1 and B[0] != None:
        Bvalue = B[0]


    # read the first point and build a root
    tempPoint = Rtreefromgithub.Point(0,allpoint[0].hilbert, allpoint[0].time)
    root = Rtreefromgithub.Leaf(Bvalue, 1, tempPoint)
    root.addChild(tempPoint)
    # add the remained points
    for i in range(1, len(allpoint)):
        tempPoint = Rtreefromgithub.Point(i,allpoint[i].hilbert, allpoint[i].time)
        insert(root, tempPoint)

    print('R-tree has been built. B is:', Bvalue, 'Highest level is:', root.level)
    return root


num = 0
block = [ ]
def Inorder(root):

    for i in range(len(root.childList)):
        if isinstance(root, Rtreefromgithub.Branch):
            Inorder(root.childList[i])
            print(root.childList[i])



def WriteData(root,allpoint):
    '''
    将第一周期的数据写到本地
    '''
    newpoint = []
    shujupoint = []

    for i in range(len(root.childList)):

        for j in range(len(root.childList[i].childList)):
                newpoint.append(root.childList[i].childList[j])
    for point in newpoint:

        shujupoint.append(allpoint[point.ident])

    file = open('SFCsforreal.dat', 'wb')
    for i in range(len(allpoint)):
            p = allpoint[i]
            b2 = struct.pack('d', p.x)
            b3 = struct.pack('d', p.y)
            b4 = struct.pack('d', p.time)
            file.write(b2)
            file.write(b3)
            file.write(b4)
    file.close()



def Writeidex(root):
    '''
    将第一周期的数据写到本地
    '''
    allleaf = []
    allleaf.append(root)
    for i in range(len(root.childList)):
        allleaf.append(root.childList[i])
    for i in range(len(root.childList)):
        for j in range(len(root.childList[i].childList)):
            allleaf.append(root.childList[i].childList[j])


    file = open('geo.dat', 'wb')
    for i in range(len(allleaf)):
        p = allleaf[i]
        p.parent = i %341
        p.blockid = i
        b1 = struct.pack('d', p.range[0])
        b2 = struct.pack('d', p.range[1])
        b3 = struct.pack('d', p.range[2])
        b4 = struct.pack('d', p.range[3])
        b5 = struct.pack('d', p.blockid)
        b6 = struct.pack('d', p.blockid)

        file.write(b1)
        file.write(b2)
        file.write(b3)
        file.write(b4)
        file.write(b5)
        file.write(b6)
    file.close()


# check the correctness of a leaf node in r-tree
def checkLeaf(leaf):
    # check whether a point is inside of a leaf
    def insideLeaf(x, y, parent):
        if x < parent[0] or x > parent[1] or y < parent[2] or y > parent[3]:
            return False
        else:
            return True

    # general check
    checkNode(leaf)
    # check whether each child point is inside of leaf's range
    for point in leaf.childList:
        if not insideLeaf(point.x, point.y, leaf.range):
            print('point(', point.x, point.y, 'is not in leaf range:', leaf.range)


# check the correctness of a branch node in r-tree
def checkBranch(branch):
    # check whether a branch is inside of another branch
    def insideBranch(child, parent):
        if child[0] < parent[0] or child[1] > parent[1] or child[2] < parent[2] or child[3] > parent[3]:
            return False
        else:
            return True

    # general check
    checkNode(branch)
    # check whether child's range is inside of this node's range
    for child in branch.childList:
        if not insideBranch(child.range, branch.range):
            print('child range:', child.range, 'is not in node range:', branch.range)
        # check this child
        if isinstance(child, Rtreefromgithub.Branch):
            # if child is still a branch node, check recursively
            checkBranch(child)
        elif isinstance(child, Rtreefromgithub.Leaf):
            # if child is a leaf node
            checkLeaf(child)


# general check for both branch and leaf node
def checkNode(node):
    global Bvalue

    length = len(node.childList)
    # check whether is empty
    if length == 0:
        print('empty node. node level:', node.level, 'node range:', node.range)
    # check whether overflow
    if length > Bvalue:
        print('overflow. node level:', node.level, 'node range:', node.range)

    # check whether the centre is really in the centre of the node's range
    r = node.range
    if (r[0] + r[1]) / 2 != node.centre[0] or (r[2] + r[3]) / 2 != node.centre[1]:
        print('wrong centre. node level:', node.level, 'node range:', node.range)
    if r[0] > r[1] or r[2] > r[3]:
        print('wrong range. node level:', node.level, 'node range:', node.range)


# read a single query from a line of text
def getQuery(nextLine):
    # split the string with whitespace
    content = nextLine.strip('\n').split(' ')
    while content.count('') != 0:
        content.remove('')
    result = []
    for s in content:
        result.append(float(s))

    return result


# sort a single query so that x1<=x1', y1<=y1'
def sortQuery(result):
    if result[0] > result[1]:
        temp = result[0]
        result[0] = result[1]
        result[1] = temp
    if result[2] > result[3]:
        temp = result[2]
        result[2] = result[3]
        result[3] = temp
    return result


# read all range queries
def readRanges(queryFile):
    fileHandle = open(queryFile, 'rt')
    queries = []
    nextLine = fileHandle.readline()

    while nextLine != '':
        query = getQuery(nextLine)
        queries.append(sortQuery(query))
        nextLine = fileHandle.readline()

    fileHandle.close()
    return queries


# determine whether a point is cover by a query(a rectangular)
def isIntersect(point, query):
    # coordinates of point
    x = point[1]
    y = point[2]
    # range of rectangular
    left = query[0]
    right = query[1]
    top = query[2]
    bottom = query[3]

    if ((x - left) * (x - right) <= 0 and (y - top) * (y - bottom) <= 0):
        return True
    else:
        return False


# answer all queries using scanning method
def scanRangeQueries(points, queries):
    # the output file
    resultFile = 'resultRange-scan.txt'
    fResult = open(resultFile, 'wt')
    # start time
    timeStart = time.time()
    # answer each query
    for query in queries:
        times = 0
        # scan each point
        for point in points:
            if isIntersect(point, query):
                times += 1

        fResult.write(str(times) + '\r\n')

    fResult.close()
    # end time
    timeEnd = time.time()
    i = len(queries)
    print('Scan range queries finished. Average time: ' + str((timeEnd - timeStart) / i))


def SFCsforreal():
    global root

    allpoint = get_NextP_Point(0, 0, 1,300000)

    get_Hilbert_For_Point_2D(allpoint, 8)
    timeStart = time.time()
    Bvalue = None
    # build r-tree from a given data-
    buildRtree(allpoint, Bvalue)
    timeEnd = time.time()
    print('build. Average time: ' + str((timeEnd - timeStart)))
    #Inorder(root)
    print(root.childList[0])

    # parse arguments
    options, args = getopt.getopt(sys.argv[1:], "d:b:")
    for opt, para in options:
        if opt == '-d':
            datasetFile = para
        if opt == '-b':
            Bvalue = int(para)
    Writeidex(root)

    #2
    for i in range(2,10):
        allpoint = get_NextP_Point(0, 0, i, 300000)
        get_Hilbert_For_Point_2D(allpoint, 8)
        timeStart = time.time()
        # INSERT r-tree from a given date
        for i in range(1, len(allpoint)):
            tempPoint = Rtreefromgithub.Point(i, allpoint[i].hilbert, allpoint[i].time)
            insert(root, tempPoint)
        timeEnd = time.time()
        print('build. Average time: ' + str((timeEnd - timeStart)))


def SFCsforsim():
    global root

    allpoint = get_NextP_Point(0, 0, 1,300000)

    get_Hilbert_For_Point_2D(allpoint, 8)
    timeStart = time.time()
    Bvalue = None
    # build r-tree from a given data-
    buildRtree(allpoint, Bvalue)
    timeEnd = time.time()
    print('build. Average time: ' + str((timeEnd - timeStart)))
    #Inorder(root)
    print(root.childList[0])

    # parse arguments
    options, args = getopt.getopt(sys.argv[1:], "d:b:")
    for opt, para in options:
        if opt == '-d':
            datasetFile = para
        if opt == '-b':
            Bvalue = int(para)
    Writeidex(root)

    #2
    for i in range(2,10):
        allpoint = get_NextP_Point(0, 0, i, 300000)
        get_Hilbert_For_Point_2D(allpoint, 8)
        timeStart = time.time()
        # INSERT r-tree from a given date
        for i in range(1, len(allpoint)):
            tempPoint = Rtreefromgithub.Point(i, allpoint[i].hilbert, allpoint[i].time)
            insert(root, tempPoint)
        timeEnd = time.time()
        print('build. Average time: ' + str((timeEnd - timeStart)))

if __name__ == "__main__":
    SFCsforsim()