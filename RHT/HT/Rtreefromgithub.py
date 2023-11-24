'''
Rtree.py
'''
import math

# r-树中的一个point
class Point:
    def __init__(self,ident, x, y):
        self.ident = ident
        self.x = x
        self.y = y

    # 获得point的位置
    def position(self, index):
        if index == 1:
            return self.x
        elif index == 2:
            return self.y


# r-树node（按叶和枝的扩展）  Bvalue为一个叶子节点最大的索引条目
class Node:
    def __init__(self, Bvalue, level):
        self.childList = []
        self.range = []
        self.centre = []
        self.Bvalue = Bvalue
        self.paren = None
        self.level = level


    # 向当前节点添加新的子节点（可以是point或node）
    def addChild(self, child):
        self.childList.append(child)
        self.update(child)

    # 添加新point或node时更新节点的覆盖范围
    def update(self, child):
        # update x range and y range
        if isinstance(child, Point):
            self.updateRange([child.x, child.x, child.y, child.y])

        elif isinstance(child, Node):
            self.updateRange(child.range)

        # update the centre coordinates  更新中心坐标
        self.centre[0] = sum(self.range[0:2]) / 2
        self.centre[1] = sum(self.range[2:4]) / 2

    # assistant function of "update" function 跟新的辅助功能
    def updateRange(self, newRange):
        # compare and update range  比较和更新范围
        if newRange[0] < self.range[0]:
            self.range[0] = newRange[0]

        if newRange[1] > self.range[1]:
            self.range[1] = newRange[1]

        if newRange[2] < self.range[2]:
            self.range[2] = newRange[2]

        if newRange[3] > self.range[3]:
            self.range[3] = newRange[3]

    # return whether the current node is overflow 当前node是否溢出
    def isOverFlow(self):
        if len(self.childList) > self.Bvalue:
            return True
        else:
            return False

    # the distance from a given point to the node centre 从给定点到节点中心的距离
    def disToCentre(self, point):
        return ((self.centre[0] - point.x) ** 2 + (self.centre[1] - point.y) ** 2) ** 0.5

    def getIncrease(self, point):
        result = 0
        # increase on x axis 在X轴上增加
        if point.x > self.range[1]:
            result += point.x - self.range[1]
        elif point.x < self.range[0]:
            result += self.range[0] - point.x
        # increase on y axis  在Y轴上增加
        if point.y > self.range[3]:
            result += point.y - self.range[3]
        elif point.y < self.range[2]:
            result += self.range[2] - point.y

        return result

    # the perimeter of current node   当前节点的边界
    def getPerimeter(self):
        return self.range[1] - self.range[0] + self.range[3] - self.range[2]

    # split a node, overridden by Leaf and Branch  拆分节点，由叶和分支覆盖
    def split(self):
        return None


# a leaf node which contains only points 只包含点的叶节点
class Leaf(Node):
    def __init__(self, Bvalue, level, point):
        super().__init__(Bvalue, level)
        self.range = [point.x, point.x, point.y, point.y]
        self.centre = [point.x, point.y]

    def split(self):
        # sort by x coordinate   按x坐标排序
        self.sortChildren(1)
        nodes = self.getBestSplit()
        periSum = nodes[0].getPerimeter() + nodes[1].getPerimeter()
        # sort by y coordinate  按y坐标排序
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
        length = len(self.childList)
        for i in range(0, length):
            for j in range(i + 1, length):
                if self.childList[i].position(index) > self.childList[j].position(index):
                    temp = self.childList[i]
                    self.childList[i] = self.childList[j]
                    self.childList[j] = temp

    # get best split based on a sorted children list  最好划分
    def getBestSplit(self):
        # used to store the minimal sum of perimeters   用于存储  边界的最小总和
        periSum = float('inf')
        # used to store the best split
        nodes = []
        b = math.floor(0.4 * self.Bvalue)
        for i in range(b, len(self.childList) - b + 1):
            # the set of the first i rectangles   第一个i矩形的集合
            node1 = Leaf(self.Bvalue, 1, self.childList[0])
            node1.paren = self.paren
            # the MBR of the first set     第一组的最小边界矩形
            for j in range(0, i):
                node1.addChild(self.childList[j])
            # the set of the remained rectangles  剩余矩形的集合
            node2 = Leaf(self.Bvalue, 1, self.childList[i])
            node2.paren = self.paren
            # the MBR of the second set
            for j in range(i, len(self.childList)):
                node2.addChild(self.childList[j])

            # check whether this is a better split
            newSum = node1.getPerimeter() + node2.getPerimeter()
            if newSum < periSum:
                periSum = newSum
                nodes = [node1, node2]

        # return the best split
        return nodes


# a branch node which contains only nodes   只包含节点的分支节点
class Branch(Node):
    def __init__(self, Bvalue , level, node):
        super().__init__(Bvalue, level)
        self.range = node.range[:]
        self.centre = node.centre[:]

    # choose a child which has a shortest distance from a given point
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

    # sort the childList by different elements of self.range
    # 按self.range的不同元素对子列表进行排序
    def sortChildren(self, index):
        length = len(self.childList)
        for i in range(0, length):
            for j in range(i + 1, length):
                if self.childList[i].range[index] > self.childList[j].range[index]:
                    temp = self.childList[i]
                    self.childList[i] = self.childList[j]
                    self.childList[j] = temp

    # get best split based on a sorted children list
    def getBestSplit(self):
        # used to store the minimal sum of perimeters
        periSum = float('inf')
        # used to store the best split
        nodes = []
        b = math.floor(0.4 * self.Bvalue)
        for i in range(b, len(self.childList) - b + 1):
            # the set of the first i rectangles
            node1 = Branch(self.Bvalue, self.level, self.childList[0])
            node1.paren = self.paren
            # the MBR of the first set
            for j in range(0, i):
                node1.addChild(self.childList[j])
            # the set of the remained rectangles
            node2 = Branch(self.Bvalue, self.level, self.childList[i])
            node2.paren = self.paren
            # the MBR of the second set
            for j in range(i, len(self.childList)):
                node2.addChild(self.childList[j])
            # check whether this is a better split
            newSum = node1.getPerimeter() + node2.getPerimeter()
            if newSum < periSum:
                periSum = newSum
                nodes = [node1, node2]
        # return the best split
        return nodes
