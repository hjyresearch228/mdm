# standard libraries
import getopt
import os
import struct as st
import time
import math
import numpy as np
import sys

# private libraries
import hilbertFromGithub as hb
import scanRange

global root
DATA = "SIM"
data_file_name = "sim"
Bvalue = 512
node_num = 0
num = 0
block_list = []
read_time = 0
query_result = []


# r-树node（按叶和枝的扩展）  Bvalue为一个叶子节点最大的索引条目
class Node:
    def __init__(self, Bvalue, level):
        self.childList = [] # 孩子列表
        self.rang = [] # 范围
        self.centre = []    # 中心
        self.Bvalue = Bvalue
        self.paren = None
        self.level = level


    # 向当前节点添加新的子节点（可以是point或node），并更新范围
    def addChild(self, child):
        self.childList.append(child)
        self.update(child)

    # 添加新point或node时更新节点的覆盖范围
    def update(self, child):
        # 更新x范围和y范围
        if isinstance(child, Point):
            self.updateRange([child.x, child.x, child.y, child.y])

        elif isinstance(child, Node):
            self.updateRange(child.rang)

        # update the centre coordinates  更新中心坐标
        self.centre[0] = sum(self.rang[0:2]) / 2
        self.centre[1] = sum(self.rang[2:4]) / 2

    # 更新函数的辅助函数
    def updateRange(self, newRange):
        # 比较并更新范围
        if newRange[0] <= self.rang[0]:
            self.rang[0] = newRange[0]

        if newRange[1] >= self.rang[1]:
            self.rang[1] = newRange[1]

        if newRange[2] <= self.rang[2]:
            self.rang[2] = newRange[2]

        if newRange[3] >= self.rang[3]:
            self.rang[3] = newRange[3]

    # 判断当前node是否溢出
    def isOverFlow(self):
        if len(self.childList) > self.Bvalue:
            return True
        else:
            return False

    # 从给定点到节点中心的距离
    def disToCentre(self, point):
        return ((self.centre[0] - point.x) ** 2 + (self.centre[1] - point.y) ** 2) ** 0.5

    def getIncrease(self, point):
        result = 0
        # increase on x axis 在X轴上增加
        if point.x > self.rang[1]:
            result += point.x - self.rang[1]
        elif point.x < self.rang[0]:
            result += self.rang[0] - point.x
        # increase on y axis  在Y轴上增加
        if point.y > self.rang[3]:
            result += point.y - self.rang[3]
        elif point.y < self.rang[2]:
            result += self.rang[2] - point.y

        return result

    # 当前节点的周长
    def getPerimeter(self):
        return self.rang[1] - self.rang[0] + self.rang[3] - self.rang[2]

    # 分裂节点, 在叶子和分支中重写
    def split(self):
        return None

    # 范围查询范围是否和节点范围重叠
    def inRange(self, rang):
        point = Point(self.rang[0], self.rang[2])
        if rang[0] <= point.x <= rang[1] and rang[2] <= point.y <= rang[3]:
            return True
        point = Point(self.rang[0], self.rang[3])
        if rang[0] <= point.x <= rang[1] and rang[2] <= point.y <= rang[3]:
            return True
        point = Point(self.rang[1], self.rang[2])
        if rang[0] <= point.x <= rang[1] and rang[2] <= point.y <= rang[3]:
            return True
        point = Point(self.rang[1], self.rang[3])
        if rang[0] <= point.x <= rang[1] and rang[2] <= point.y <= rang[3]:
            return True
        point = Point(rang[0], rang[2])
        if self.rang[0] <= point.x <= self.rang[1] and self.rang[2] <= point.y <= self.rang[3]:
            return True
        point = Point(rang[0], rang[3])
        if self.rang[0] <= point.x <= self.rang[1] and self.rang[2] <= point.y <= self.rang[3]:
            return True
        point = Point(rang[1], rang[2])
        if self.rang[0] <= point.x <= self.rang[1] and self.rang[2] <= point.y <= self.rang[3]:
            return True
        point = Point(rang[1], rang[3])
        if self.rang[0] <= point.x <= self.rang[1] and self.rang[2] <= point.y <= self.rang[3]:
            return True
        if self.rang[0] <= rang[0] and self.rang[1] >= rang[1] and rang[2] <= self.rang[2] and rang[3] >= self.rang[3]:
            return True
        if rang[0] <= self.rang[0] and rang[1] >= self.rang[1] and self.rang[2] <= rang[2] and self.rang[3] >= rang[3]:
            return True
        return False


# r-树中的一个point
class Point:
    def __init__(self,x, y):
        self.x = x
        self.y = y

    # 获得point的位置
    def position(self, index):
        if index == 1:
            return self.x
        elif index == 2:
            return self.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.x) * hash(self.y)


# 读取数据集
def read_data(file_path):
    point_list = []
    dirs = os.listdir(file_path)
    for file_name in dirs:
        with open(file_path + "/" + file_name) as file_object:
            for line in file_object:
                line = line.rstrip()
                line = line.split(",")
                p = Point(float(line[3]), float(line[4]))
                point_list.append(p)
    return point_list


#节点溢出
def handleOverFlow(node):
    global root
    global Bvalue
    # 分裂一个节点为两个新节点
    nodes = node.split()
    # 如果根节点溢出，需要建立新的根
    if node.paren == None:
        root = Branch(Bvalue, node.level + 1, nodes[0])
        root.addChild(nodes[0])
        root.addChild(nodes[1])
        root.childList[0].paren = root
        root.childList[1].paren = root
    else:
        # 更新父节点
        parent = node.paren
        parent.childList.remove(node)
        parent.childList += nodes
        # 检查父节点是否溢出
        if parent.isOverFlow():
            handleOverFlow(parent)


# a branch node which contains only nodes   只包含节点的分支节点
class Branch(Node):
    def __init__(self, Bvalue , level, node):
        super().__init__(Bvalue, level)
        self.rang = node.rang[:]
        self.centre = node.centre[:]

    # 选择一个与 给定point 距离最短的孩子
    def chooseChild(self, point):
        result = None
        increase = None
        for child in self.childList:
            newIncrease = child.disToCentre(point)
            # newIncrease = child.getIncrease(point)
            if increase == None:
                increase = newIncrease
                result = child
            elif increase != 0 and newIncrease / increase > 0.93 and newIncrease / increase < 1.07:
                if len(result.childList) / len(child.childList) > 2:
                    increase = newIncrease
                    result = child
            elif newIncrease < increase:
                increase = newIncrease
                result = child

        return result

    def split(self):
        # sort by xleft and get the sum of perimeter  按xleft排序，得到边界之和
        self.sortChildren(0)
        nodes = self.getBestSplit()
        periSum = nodes[0].getPerimeter() + nodes[1].getPerimeter()
        # sort by xright, ybottom, ytop respectively   分别按xright、ybottom和ytop排序
        for i in range(1, 4):
            self.sortChildren(i)
            newNodes = self.getBestSplit()
            newSum = newNodes[0].getPerimeter() + newNodes[1].getPerimeter()
            # check whether this is a better split  检查这是否是一个更好的分割
            if newSum < periSum:
                periSum = newSum
                nodes = newNodes

        # set nodes parents and return the best split
        # 设置节点父节点并返回最佳分割
        for node in nodes[0].childList:
            node.paren = nodes[0]
        for node in nodes[1].childList:
            node.paren = nodes[1]
        return nodes

    # sort the childList by different elements of self.rang
    # 按self.rang的不同元素对子列表进行排序
    def sortChildren(self, index):
        if index == 0:
            self.childList.sort(key=lambda x:x.rang[0])
        elif index == 1:
            self.childList.sort(key=lambda x:x.rang[1])
        elif index == 2:
            self.childList.sort(key=lambda x:x.rang[2])
        elif index == 3:
            self.childList.sort(key=lambda x:x.rang[3])
        # length = len(self.childList)
        # 冒泡排序
        # for i in range(0, length):
        #     for j in range(i + 1, length):
        #         if self.childList[i].rang[index] > self.childList[j].rang[index]:
        #             temp = self.childList[i]
        #             self.childList[i] = self.childList[j]
        #             self.childList[j] = temp

    # # get best split based on a sorted children list
    # def getBestSplit(self):
    #     # used to store the minimal sum of perimeters
    #     periSum = float('inf')
    #     # used to store the best split
    #     nodes = []
    #     b = math.floor(0.5 * self.Bvalue)
    #     for i in range(b, len(self.childList) - b + 1):
    #         # the set of the first i rectangles
    #         node1 = Branch(self.Bvalue, self.level, self.childList[0])
    #         node1.paren = self.paren
    #         # the MBR of the first set
    #         for j in range(0, i):
    #             node1.addChild(self.childList[j])
    #         # the set of the remained rectangles
    #         node2 = Branch(self.Bvalue, self.level, self.childList[i])
    #         node2.paren = self.paren
    #         # the MBR of the second set
    #         for j in range(i, len(self.childList)):
    #             node2.addChild(self.childList[j])
    #         # check whether this is a better split
    #         newSum = node1.getPerimeter() + node2.getPerimeter()
    #         if newSum < periSum:
    #             periSum = newSum
    #             nodes = [node1, node2]
    #     # return the best split
    #     return nodes

    # get best split based on a sorted children list
    def getBestSplit(self):
        # used to store the minimal sum of perimeters
        # periSum = float('inf')
        # used to store the best split
        nodes = []
        b = math.floor(0.5 * self.Bvalue)
        # for i in range(b, len(self.childList) - b + 1):
        # the set of the first i rectangles
        node1 = Branch(self.Bvalue, self.level, self.childList[0])
        node1.paren = self.paren
        # the MBR of the first set
        for j in range(0, b):
            node1.addChild(self.childList[j])
        # the set of the remained rectangles
        node2 = Branch(self.Bvalue, self.level, self.childList[b])
        node2.paren = self.paren
        # the MBR of the second set
        for j in range(b, len(self.childList)):
            node2.addChild(self.childList[j])
        # check whether this is a better split
        # newSum = node1.getPerimeter() + node2.getPerimeter()
        # if newSum < periSum:
        #     periSum = newSum
        #     nodes = [node1, node2]
        nodes = [node1, node2]
        # return the best split
        return nodes



# 将点插入到节点中
def insert(node, point):
    # 如果节点是叶子，则添加这个点
    if isinstance(node, Leaf):
        node.addChild(point)
        if node.isOverFlow():
            handleOverFlow(node)
    # 如果节点是分支，则选择一个孩子添加这个点
    elif isinstance(node, Branch):
        node.update(point)
        childNode = node.chooseChild(point)
        insert(childNode, point)
    else:
        pass


# 只包含点的叶子节点
class Leaf(Node):
    id = -1
    def __init__(self, Bvalue, level, point):
        super().__init__(Bvalue, level)
        self.rang = [point.x, point.x, point.y, point.y]
        self.centre = [point.x, point.y]

    def split(self):
        # 按x坐标排序
        self.sortChildren(1)
        nodes = self.getBestSplit()
        periSum = nodes[0].getPerimeter() + nodes[1].getPerimeter()
        # 按y坐标排序
        self.sortChildren(2)
        newNodes = self.getBestSplit()
        newSum = newNodes[0].getPerimeter() + newNodes[1].getPerimeter()
        # return the best split  返回最佳分割
        if newSum < periSum:
            return newNodes
        else:
            return nodes

    # sort the childList by x if index is 1, by y if index is 2  排序子列表
    def sortChildren(self, index):
        if index == 1:
            self.childList.sort(key=lambda p:p.x)
        elif index == 2:
            self.childList.sort(key=lambda p:p.y)
        # length = len(self.childList)
        # 冒泡排序
        # for i in range(0, length):
        #     for j in range(i + 1, length):
        #         if self.childList[i].position(index) > self.childList[j].position(index):
        #             temp = self.childList[i]
        #             self.childList[i] = self.childList[j]
        #             self.childList[j] = temp

    # # get best split based on a sorted children list  最好划分
    # def getBestSplit(self):
    #     # used to store the minimal sum of perimeters   用于存储  边界的最小总和
    #     periSum = float('inf')
    #     # used to store the best split
    #     nodes = []
    #     b = math.floor(0.4 * self.Bvalue)
    #     for i in range(b, len(self.childList) - b + 1):
    #         # the set of the first i rectangles   第一个i矩形的集合
    #         node1 = Leaf(self.Bvalue, 1, self.childList[0])
    #         node1.paren = self.paren
    #         # the MBR of the first set     第一组的最小边界矩形
    #         for j in range(0, i):
    #             node1.addChild(self.childList[j])
    #         # the set of the remained rectangles  剩余矩形的集合
    #         node2 = Leaf(self.Bvalue, 1, self.childList[i])
    #         node2.paren = self.paren
    #         # the MBR of the second set
    #         for j in range(i, len(self.childList)):
    #             node2.addChild(self.childList[j])
    #         # check whether this is a better split
    #         newSum = node1.getPerimeter() + node2.getPerimeter()
    #         if newSum < periSum:
    #             periSum = newSum
    #             nodes = [node1, node2]
    #
    #     # return the best split
    #     return nodes

    # get best split based on a sorted children list  最好划分
    def getBestSplit(self):
        # used to store the minimal sum of perimeters   用于存储  边界的最小总和
        # periSum = float('inf')
        # used to store the best split
        nodes = []
        b = math.floor(0.5 * self.Bvalue)
        # the set of the first i rectangles   第一个i矩形的集合
        node1 = Leaf(self.Bvalue, 1, self.childList[0])
        node1.paren = self.paren
        # the MBR of the first set     第一组的最小边界矩形
        for j in range(0, b):
            node1.addChild(self.childList[j])
        # the set of the remained rectangles  剩余矩形的集合
        node2 = Leaf(self.Bvalue, 1, self.childList[b])
        node2.paren = self.paren
        # the MBR of the second set
        for j in range(b, len(self.childList)):
            node2.addChild(self.childList[j])
        # check whether this is a better split
        # newSum = node1.getPerimeter() + node2.getPerimeter()
        # if newSum < periSum:
        #     periSum = newSum
        #     nodes = [node1, node2]
        nodes = [node1, node2]
        # return the best split
        return nodes


# 构建R树
def buildRtree(allpoint):
    global root
    global Bvalue
    # 读取第一个点，建立一个根
    tempPoint = allpoint[0]
    root = Leaf(Bvalue, 1, tempPoint)
    root.addChild(tempPoint)
    # 将剩下的点加入
    for i in range(1, len(allpoint)):
        insert(root, allpoint[i])
    print('R-tree has been built. B is:', Bvalue, 'Highest level is:', root.level)
    return root


# 中序遍历
def Inorder(root):
    global num
    global block_list
    global node_num
    node_num += 1
    if isinstance(root, Branch):
        for i in range(len(root.childList)):
            Inorder(root.childList[i])
    elif isinstance(root, Leaf):
        root.id = num
        num += 1
        block_list.append(root.childList)

# 将块写入硬盘
def write_block(arr, data_file):
    for i in range(len(arr)):
        block = arr[i]
        data_file.seek(i * (512 * 16))
        for p in block:
            b_x = st.pack("d", p.x)
            b_y = st.pack("d", p.y)
            data_file.write(b_x)
            data_file.write(b_y)
        for i in range(512 - len(arr)):
            data_file.write(st.pack("d", 0))
            data_file.write(st.pack("d", 0))

# 将块从磁盘中读出
def read_block(id, data_file):
    point_list = []
    data_file.seek(id * 8 * 1024)
    data_str = data_file.read(8 * 1024)
    for i in range(512):
        if len(data_str[i * 16:i * 16 + 16]) != 16:
            break
        data_tuple = st.unpack("dd", data_str[i * 16:i * 16 + 16])
        if data_tuple[0] == 0 and data_tuple[1] == 0:
            break
        p = Point(float(data_tuple[0]), float(data_tuple[1]))
        point_list.append(p)
    return point_list


# range query
def range_query(rang, root, data_file):
    global read_time
    if isinstance(root, Branch):
        for child in root.childList:
            # if isinstance(child, Branch):
            #     range_query(rang, child, data_file)
            # elif isinstance(child, Leaf):
            # 如果节点范围和范围有重叠部分
            if child.inRange(rang):
                range_query(rang, child, data_file)
    elif isinstance(root, Leaf):
        read_time += 1
        point_list = read_block(root.id, data_file)
        for p in point_list:
            if rang[0] <= p.x <= rang[1] and rang[2] <= p.y <= rang[3]:
                query_result.append(p)


def HT_Sim():
    global node_num
    ep = 1
    all_time = 0
    time_start = time.time()
    allpoint = []
    # 读取数据
    file_path = "../" + DATA + "/EP" + str(ep) + "/"
    allpoint += read_data(file_path)
    allpoint = list(set(allpoint))
    print(len(allpoint))

    # 构建R树
    root = buildRtree(allpoint)
    Inorder(root)
    data_file = open(DATA + "/models/EP" + str(ep) + "/" + data_file_name + ".dat", "wb")
    write_block(block_list, data_file)
    data_file.close()
    time_end = time.time()
    print("构建时间:", str(time_end - time_start), " s")
    print("节点大小" + str(sys.getsizeof(root)) + "B")
    print("节点个数" + str(node_num))

    if ep <= 2:
        # 范围查询
        time_start = time.time()
        point_sum = 0
        ranges = []
        data_file = open(DATA+"/models/EP"+str(ep)+"/"+data_file_name+".dat", "rb")
        with open("../"+DATA+"/range_"+data_file_name+".txt") as query_file:
            for line in query_file:
                line = line.rstrip()
                line = line.split(",")
                rang = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
                ranges.append(rang)
        if ep == 1:
            for rang in ranges[:250]:
                range_query(rang, root, data_file)
        if ep == 2:
            for rang in ranges[250:]:
                range_query(rang, root, data_file)
        data_file.close()
        time_end = time.time()
        query_time = time_end - time_start
        print("查询到的点个数："+str(len(query_result)))
        print("查询时间为:" + str(query_time))
        print("读取块数：" + str(read_time))

    # # 范围查询
    # time_start = time.time()
    # point_sum = 0
    # read_time = 0
    # ranges1 = []
    # ranges = []
    # data_file = open(DATA + "/models/EP" + str(ep) + "/" + data_file_name + ".dat", "rb")
    # with open("../" + DATA + "/range_" + data_file_name + ".txt") as query_file:
    #     for line in query_file:
    #         line = line.rstrip()
    #         line = line.split(",")
    #         rang = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
    #         ranges1.append(rang)
    #     ranges.extend(ranges1[:25])
    #     ranges.extend(ranges1[125:150])
    #     ranges.extend(ranges1[250:275])
    #     ranges.extend(ranges1[375:400])
    #     print(len(ranges))
    #     for rang in ranges:
    #         result = []
    #         range_query(rang, root, data_file)
    #         point_sum += len(result)
    # data_file.close()
    # time_end = time.time()
    # query_time = time_end - time_start
    # print("查询到的点个数：" + str(point_sum))
    # print("查询时间为:" + str(query_time))
    # print("读取块数：" + str(read_time))

if __name__ == "__main__":
    HT_Sim()