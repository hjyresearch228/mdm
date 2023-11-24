import os
import struct
import time
import sys
import random


class point:
    def __init__(self):
        self.pos = None  # 位置
        self.time = None  # 时间

class region:
    def __init__(self):
        # sim数据集范围
        self.pos_min = 0.0
        self.pos_max = 6884.0
        self.time_min = 0.0
        self.time_max = 999.0
        # geo数据集范围
        # self.pos_min = 0.0
        # self.pos_max=63576.0
        # self.time_min = 0.0
        # self.time_max = 2678400.0

# 结点
class Node:
    def __init__(self):
        self.flag = 0  # 区域页面的标记，0为点页面，1为区域页面
        self.r = region()  # 该页面的区域
        self.pageList = []  # 列表，区域页面用于存放pageID和region，点页面用于存放数据点坐标
        self.parentID = -1  # 父页面id

class KDBTree:
    nodes = []  # 用于存放页面结点
    rootID = 0  # 根页面ID

class Block:
    block_size=512  # 数据块容量
    block_index=227  # 索引块容量

class P:
    index_path='..//KDB//sim.dix'
    data_path='..//KDB//sim.dat'
    block_access=0

def inRegion(region1, region2):
    '''
    判断region1是否包含region2
    :param region1: 区域1
    :param region2: 区域2
    :return: Bool类型
    '''
    if (region1.pos_min <= region2.pos_min) and (region1.pos_max >= region2.pos_max) and (region1.time_min <= region2.time_min) and (region1.time_max >= region2.time_max):
        return True
    else:
        return False


def p_pageSplit(t, pageID, xi, i, leftPage, rightPage):
    '''
    点页面的分裂
    :param t: KDBTree
    :param pageID: 分裂的页面id
    :param xi: 分裂点
    :param i: 分裂维度
    :param leftPage: 分裂产生的左页面
    :param rightPage: 分裂产生的右页面
    '''
    if (i == 0) and (xi >= t.nodes[pageID].r.pos_min) and (xi <= t.nodes[pageID].r.pos_max):
        t_r = region()
        t_r.pos_min = t.nodes[pageID].r.pos_min
        t_r.pos_max = xi
        t_r.time_min = t.nodes[pageID].r.time_min
        t_r.time_max = t.nodes[pageID].r.time_max
        leftPage.r = t_r
        t_r1 = region()
        t_r1.pos_min = xi
        t_r1.pos_max = t.nodes[pageID].r.pos_max
        t_r1.time_min = t.nodes[pageID].r.time_min
        t_r1.time_max = t.nodes[pageID].r.time_max
        rightPage.r = t_r1
        for i in range(len(t.nodes[pageID].pageList)):
            if t.nodes[pageID].pageList[i].pos <= xi:
                leftPage.pageList.append(t.nodes[pageID].pageList[i])
            else:
                rightPage.pageList.append(t.nodes[pageID].pageList[i])
    elif (i == 1) and (xi >= t.nodes[pageID].r.time_min) and (xi <= t.nodes[pageID].r.time_max):
        t_r = region()
        t_r.pos_min = t.nodes[pageID].r.pos_min
        t_r.pos_max = t.nodes[pageID].r.pos_max
        t_r.time_min = t.nodes[pageID].r.time_min
        t_r.time_max = xi
        leftPage.r = t_r
        t_r1 = region()
        t_r1.pos_min = t.nodes[pageID].r.pos_min
        t_r1.pos_max = t.nodes[pageID].r.pos_max
        t_r1.time_min = xi
        t_r1.time_max = t.nodes[pageID].r.time_max
        rightPage.r = t_r1
        for p in t.nodes[pageID].pageList:
            if p.time <= xi:
                leftPage.pageList.append(p)
            else:
                rightPage.pageList.append(p)
    else:
        print("点页面分裂失败")


def r_pageSplit(t, pageID, xi, i, leftPage, rightPage):
    '''
    区域页面分裂
    :param t: KDBTree
    :param pageID: 分裂的页面id
    :param xi: 分裂点
    :param i: 分裂维度
    :param leftPage: 分裂产生的左页面
    :param rightPage: 分裂产生的右页面
    '''
    t.nodes.append(rightPage)
    if (i == 0) and (xi > t.nodes[pageID].r.pos_min) and (xi < t.nodes[pageID].r.pos_max):
        t_r = region()
        t_r.pos_min = t.nodes[pageID].r.pos_min
        t_r.pos_max = xi
        t_r.time_min = t.nodes[pageID].r.time_min
        t_r.time_max = t.nodes[pageID].r.time_max
        leftPage.r = t_r
        t_r1 = region()
        t_r1.pos_min = xi
        t_r1.pos_max = t.nodes[pageID].r.pos_max
        t_r1.time_min = t.nodes[pageID].r.time_min
        t_r1.time_max = t.nodes[pageID].r.time_max
        rightPage.r = t_r1
        for re in t.nodes[pageID].pageList:
            if inRegion(leftPage.r, re[1]):
                leftPage.pageList.append(re)
                t.nodes[re[0]].parentID = pageID
            elif inRegion(rightPage.r, re[1]):
                rightPage.pageList.append(re)
                t.nodes[re[0]].parentID = t.nodes.index(rightPage)
            else:
                l_page = Node()  # 新产生的左页面
                r_page = Node()  # 新产生的右页面
                if (t.nodes[re[0]].flag == 0):  # 如果该页面是点页面，则进行点页面分裂
                    p_pageSplit(t, re[0], xi, i, l_page, r_page)
                    l_page.parentID = pageID
                    r_page.parentID = t.nodes.index(rightPage)
                    t.nodes[re[0]] = l_page
                    leftPage.pageList.append([re[0], l_page.r])
                    t.nodes.append(r_page)
                    rightPage.pageList.append([len(t.nodes) - 1, r_page.r])

                elif (t.nodes[re[0]].flag == 1):  # 区域页面则递归
                    r_pageSplit(t, re[0], xi, i, l_page, r_page)
                    l_page.flag = 1
                    r_page.flag = 1
                    l_page.parentID = pageID
                    r_page.parentID = t.nodes.index(rightPage)
                    t.nodes[re[0]] = l_page
                    leftPage.pageList.append([re[0], l_page.r])
                    rightPage.pageList.append([t.nodes.index(r_page), r_page.r])
    elif (i == 1) and (xi > t.nodes[pageID].r.time_min) and (xi < t.nodes[pageID].r.time_max):
        t_r = region()
        t_r.pos_min = t.nodes[pageID].r.pos_min
        t_r.pos_max = t.nodes[pageID].r.pos_max
        t_r.time_min = t.nodes[pageID].r.time_min
        t_r.time_max = xi
        leftPage.r = t_r
        t_r1 = region()
        t_r1.pos_min = t.nodes[pageID].r.pos_min
        t_r1.pos_max = t.nodes[pageID].r.pos_max
        t_r1.time_min = xi
        t_r1.time_max = t.nodes[pageID].r.time_max
        rightPage.r = t_r1
        for re in t.nodes[pageID].pageList:
            if inRegion(leftPage.r, re[1]):
                leftPage.pageList.append(re)
                t.nodes[re[0]].parentID = pageID
            elif inRegion(rightPage.r, re[1]):
                rightPage.pageList.append(re)
                t.nodes[re[0]].parentID = t.nodes.index(rightPage)
            else:
                l_page = Node()  # 新产生的左页面
                r_page = Node()  # 新产生的右页面
                if (t.nodes[re[0]].flag == 0):  # 如果该页面是点页面，则进行点页面分裂
                    p_pageSplit(t, re[0], xi, i, l_page, r_page)
                    l_page.parentID = pageID
                    r_page.parentID = t.nodes.index(rightPage)
                    t.nodes[re[0]] = l_page
                    leftPage.pageList.append([re[0], l_page.r])
                    t.nodes.append(r_page)
                    rightPage.pageList.append([len(t.nodes) - 1, r_page.r])
                elif (t.nodes[re[0]].flag == 1):  # 区域页面则递归
                    r_pageSplit(t, re[0], xi, i, l_page, r_page)
                    l_page.flag = 1
                    r_page.flag = 1
                    l_page.parentID = pageID
                    r_page.parentID = t.nodes.index(rightPage)
                    t.nodes[re[0]] = l_page
                    leftPage.pageList.append([re[0], l_page.r])
                    rightPage.pageList.append([t.nodes.index(r_page), r_page.r])
    else:
        print("区域页面分裂失败")


def point_query(t, pageId, p, p_page):
    '''
    点查询
    :param t: KDBTree
    :param pageId: 当前要查询的页面id
    :param p: 要查询的轨迹点
    :param p_page:匹配到的页面的父页面
    :return: 父页面id,匹配到的页面id,查询到点的标记
    '''
    flag = [0, ]
    parent_page = p_page
    pageId = pageId
    if getattr(t.nodes[pageId], 'flag') == 1:  # 区域页面
        temp = getattr(t.nodes[pageId], 'pageList')
        for j in temp:
            if (p.pos >= j[1].pos_min) and (p.pos <= j[1].pos_max) and (p.time >= j[1].time_min) and (p.time <= j[1].time_max):
                parent_page = pageId
                return point_query(t, j[0], p, parent_page)
        return -1, -1, flag
    else:
        temp = getattr(t.nodes[pageId], 'pageList')
        for i in temp:
            if (i.pos == p.pos) and (i.time == p.time):
                flag.append(1)
                break
        return parent_page, pageId, flag


height = 0  # 树的高度
split_flag = [0, ]#分裂标记
def insert_kdb(t, p):
    '''
    插入操作
    :param t: KDBTree
    :param p: 插入的点
    '''
    global height
    if len(t.nodes) == 0:  # 树为空
        pointPage = Node()  # 创建点页面
        pointPage.pageList.append(p)
        t.nodes.append(pointPage)
        t.rootID = t.nodes.index(pointPage)
        height += 1
    else:
        parentPageId, pointPageId, fl = point_query(t, t.rootID, p,
                                                    t.rootID)  # parentPage,pointPage分别为父页面和匹配到的点页面,flag用于判断是否查询到点
        if (parentPageId != -1) and (pointPageId != -1):
            if len(fl) == 2:
                print("插入失败，该点已存在!\n")
            else:
                pointPage = t.nodes[pointPageId]
                pointPage.pageList.append(p)
                if len(pointPage.pageList) > Block.block_size:  # 点页面溢出
                    lPage = Node()
                    rPage = Node()
                    x=0
                    if split_flag[0] == 0:
                        x_l = []
                        for i in pointPage.pageList:
                            x_l.append(i.pos)
                        x = (min(x_l) + max(x_l)) / 2
                    elif split_flag[0] == 1:
                        x_l = []
                        for i in pointPage.pageList:
                            x_l.append(i.time)
                        x = (min(x_l) + max(x_l)) / 2
                    p_pageSplit(t, pointPageId, x, split_flag[0], lPage, rPage)
                    split_flag[0] = (split_flag[0] + 1) % 2
                    if pointPageId == t.rootID:
                        rootPage = Node()  # 创建根页面 即区域页面
                        rootPage.flag = 1
                        t.nodes.append(rootPage)
                        lPage.parentID = t.nodes.index(rootPage)
                        rPage.parentID = t.nodes.index(rootPage)
                        t.nodes[pointPageId] = lPage
                        t.nodes.append(rPage)
                        rootPage.pageList.append([t.nodes.index(lPage), lPage.r])
                        rootPage.pageList.append([t.nodes.index(rPage), rPage.r])
                        t.rootID = t.nodes.index(rootPage)
                        height += 1
                    else:
                        lPage.parentID = parentPageId
                        rPage.parentID = parentPageId
                        t.nodes[pointPageId] = lPage
                        t.nodes.append(rPage)
                        for i in t.nodes[parentPageId].pageList:
                            if i[0] == pointPageId:
                                i[1] = lPage.r
                        t.nodes[parentPageId].pageList.append([t.nodes.index(rPage), rPage.r])
                        if len(t.nodes[parentPageId].pageList) > Block.block_index:  # 区域页面溢出
                            left_page = Node()
                            right_page = Node()
                            split_x=0
                            if split_flag[0] == 0:
                                split_x = (t.nodes[parentPageId].r.pos_min + t.nodes[parentPageId].r.pos_max) / 2
                            elif split_flag[0] == 1:
                                split_x = (t.nodes[parentPageId].r.time_min + t.nodes[parentPageId].r.time_max) / 2
                            r_pageSplit(t, parentPageId, split_x, split_flag[0], left_page, right_page)
                            split_flag[0] = (split_flag[0] + 1) % 2
                            left_page.flag = 1
                            right_page.flag = 1
                            if parentPageId == t.rootID:
                                rootPage = Node()  # 创建根页面 即区域页面
                                rootPage.flag = 1
                                t.nodes.append(rootPage)
                                left_page.parentID = t.nodes.index(rootPage)
                                right_page.parentID = t.nodes.index(rootPage)
                                t.nodes[parentPageId] = left_page
                                rootPage.pageList.append([t.nodes.index(left_page), left_page.r])
                                rootPage.pageList.append([t.nodes.index(right_page), right_page.r])
                                t.rootID = t.nodes.index(rootPage)
                                height += 1
                            else:
                                left_page.parentID = t.nodes[parentPageId].parentID
                                right_page.parentID = t.nodes[parentPageId].parentID
                                t.nodes[parentPageId] = left_page
                                for i in t.nodes[left_page.parentID].pageList:
                                    if i[0] == parentPageId:
                                        i[1] = left_page.r
                                t.nodes[left_page.parentID].pageList.append([t.nodes.index(right_page), right_page.r])
                                parentPageId1 = []
                                parentPageId1.append(left_page.parentID)
                                while (parentPageId1[0] != t.rootID):
                                    if len(t.nodes[parentPageId1[0]].pageList) > Block.block_index:  # 区域页面溢出
                                        left_page = Node()
                                        right_page = Node()
                                        split_x1=0
                                        if split_flag[0] == 0:
                                            split_x1 = (t.nodes[parentPageId1[0]].r.pos_min + t.nodes[
                                                parentPageId1[0]].r.pos_max) / 2
                                        elif split_flag[0] == 1:
                                            split_x1 = (t.nodes[parentPageId1[0]].r.time_min + t.nodes[
                                                parentPageId1[0]].r.time_max) / 2
                                        r_pageSplit(t, parentPageId1[0], split_x1, split_flag[0], left_page, right_page)
                                        split_flag[0] = (split_flag[0] + 1) % 2
                                        left_page.flag = 1
                                        right_page.flag = 1
                                        left_page.parentID = t.nodes[parentPageId1[0]].parentID
                                        right_page.parentID = t.nodes[parentPageId1[0]].parentID
                                        t.nodes[parentPageId1[0]] = left_page
                                        for i in t.nodes[left_page.parentID].pageList:
                                            if i[0] == parentPageId1[0]:
                                                i[1] = left_page.r
                                        t.nodes[left_page.parentID].pageList.append(
                                            [t.nodes.index(right_page), right_page.r])
                                        parentPageId1[0] = left_page.parentID
                                    else:
                                        break
                                if parentPageId1[0] == t.rootID:
                                    if len(t.nodes[parentPageId1[0]].pageList) > Block.block_index:  # 区域页面溢出
                                        left_page = Node()
                                        right_page = Node()
                                        split_x2=0
                                        if split_flag[0] == 0:
                                            split_x2 = (t.nodes[parentPageId1[0]].r.pos_min + t.nodes[
                                                parentPageId1[0]].r.pos_max) / 2
                                        elif split_flag[0] == 1:
                                            split_x2 = (t.nodes[parentPageId1[0]].r.time_min + t.nodes[
                                                parentPageId1[0]].r.time_max) / 2
                                        r_pageSplit(t, parentPageId1[0], split_x2, split_flag[0], left_page, right_page)
                                        split_flag[0] = (split_flag[0] + 1) % 2
                                        left_page.flag = 1
                                        right_page.flag = 1
                                        rootPage = Node()  # 创建根页面 即区域页面
                                        rootPage.flag = 1
                                        t.nodes.append(rootPage)
                                        left_page.parentID = t.nodes.index(rootPage)
                                        right_page.parentID = t.nodes.index(rootPage)
                                        t.nodes[parentPageId1[0]] = left_page
                                        rootPage.pageList.append([t.nodes.index(left_page), left_page.r])
                                        rootPage.pageList.append([t.nodes.index(right_page), right_page.r])
                                        t.rootID = t.nodes.index(rootPage)
                                        height += 1


def read_datasets(path, datasets):
    '''
    读取数据集
    :param path: 数据存放的路径
    :param datasets: 存放读取到的轨迹点
    :param pe: 周期
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
        p=point()
        p.pos = float(splitData[j][3])
        p.time = float(splitData[j][4])
        datasets.append(p)


def write_data(t):
    # 写数据
    dataIdDict = {}  # 键为数据块号，值为点页面id
    id = 0  # 块号
    file = open(P.data_path, 'wb')
    for no in t.nodes:
        if no.flag == 0:
            dataIdDict[id] = t.nodes.index(no)
            id += 1
            for j in no.pageList:
                b1 = struct.pack('d', j.pos)
                b2 = struct.pack('d', j.time)
                file.write(b1)
                file.write(b2)
            for j in range(Block.block_size - len(no.pageList)):
                x = -1
                y = -1
                b1 = struct.pack('d', x)
                b2 = struct.pack('d', y)
                file.write(b1)
                file.write(b2)
    file.close()
    # 写索引
    indexIdDict = {}  # 键为索引块号，值为区域页面id
    id1 = 0  # 块号
    file1 = open(P.index_path, 'wb')
    for no in t.nodes:
        if no.flag == 1:
            indexIdDict[id1] = t.nodes.index(no)
            id1 += 1
            for j in no.pageList:
                b1 = struct.pack('i', j[0])
                b2 = struct.pack('d', j[1].pos_min)
                b3 = struct.pack('d', j[1].pos_max)
                b4 = struct.pack('d', j[1].time_min)
                b5 = struct.pack('d', j[1].time_max)
                file1.write(b1)
                file1.write(b2)
                file1.write(b3)
                file1.write(b4)
                file1.write(b5)
            for j in range(Block.block_index - len(no.pageList)):
                b1 = struct.pack('i', -1)
                b2 = struct.pack('d', -1)
                b3 = struct.pack('d', -1)
                b4 = struct.pack('d', -1)
                b5 = struct.pack('d', -1)
                file1.write(b1)
                file1.write(b2)
                file1.write(b3)
                file1.write(b4)
                file1.write(b5)
    file1.close()
    return dataIdDict, indexIdDict


def hasInter(r1, r2):
    '''
    判断两个区域是否有交集
    '''
    if (r1.pos_max <= r2.pos_min) or (r1.pos_min >= r2.pos_max) or (r1.time_max <= r2.time_min) or (r1.time_min >= r2.time_max):
        return False
    else:
        return True


def region_query(pageId, _r, resultsId, index_buff, indexDict):
    '''
    :param pageId: 要查询的当前页面id
    :param _r: 要查询的范围
    :param resultsId: 匹配到的页面id,集合
    :param index_buff: 存放索引的缓冲区
    :param indexDict: 字典，键是块号，值是页面号
    '''
    if pageId in index_buff:
        for j in index_buff[pageId].pageList:
            if hasInter(j[1], _r) == True:
                region_query(j[0], _r, resultsId, index_buff, indexDict)
    elif len(index_buff) < 128 and (pageId in indexDict.values()):
        file = open(P.index_path, 'rb')
        P.block_access += 1
        for key, value in indexDict.items():
            if value == pageId:
                file.seek(key * 8172)
                count = 0
                s = file.read(8172)
                n = Node()
                n.flag = 1
                val = indexDict[key]  # 页面号
                while count < Block.block_index:
                    id = struct.unpack('i', s[count * 36:count * 36 + 4])
                    ss = struct.unpack('dddd', s[count * 36+ 4:count * 36 + 36])
                    if id[0] == -1:
                        break
                    r = region()
                    r.pos_min = ss[0]
                    r.pos_max = ss[1]
                    r.time_min = ss[2]
                    r.time_max = ss[3]
                    n.pageList.append([id[0], r])
                    count += 1
                index_buff[val] = n
                file.close()
                for j in index_buff[pageId].pageList:
                    if hasInter(j[1], _r) == True:
                        region_query(j[0], _r, resultsId, index_buff, indexDict)
                break
    elif len(index_buff) == 128 and (pageId in indexDict.values()):  # 缓存已满
        k = list(index_buff.keys())
        del index_buff[k[0]]
        file = open(P.index_path, 'rb')
        P.block_access += 1
        for key, value in indexDict.items():
            if value == pageId:
                file.seek(key * 8172)
                count = 0
                s = file.read(8172)
                n = Node()
                n.flag = 1
                val = indexDict[key]  # 页面号
                while count < Block.block_index:
                    id = struct.unpack('i', s[count * 36:count * 36 + 4])
                    ss = struct.unpack('dddd', s[count * 36 + 4:count * 36 + 36])
                    if id[0] == -1:
                        break
                    r = region()
                    r.pos_min = ss[0]
                    r.pos_max = ss[1]
                    r.time_min = ss[2]
                    r.time_max = ss[3]
                    n.pageList.append([id[0], r])
                    count += 1
                index_buff[val] = n
                file.close()
                for j in index_buff[pageId].pageList:
                    if hasInter(j[1], _r) == True:
                        region_query(j[0], _r, resultsId, index_buff, indexDict)
                break
    else:
        resultsId.add(pageId)


def range_filter(resultsId, _r, data_buff, dataDict):
    '''
    :param resultsId: 要查询的点页面id
    :param _r: 范围
    :param data_buff: 存放数据的缓冲区
    :param dataDict: 字典，键是块号，值是页面号
    :return: 查询到的点数
    '''
    recall = 0
    for pageId in resultsId:
        if pageId in data_buff:
            for i in data_buff[pageId].pageList:
                if i.pos >= _r.pos_min and i.pos <= _r.pos_max and i.time >= _r.time_min and i.time <= _r.time_max:
                    recall += 1
        elif len(data_buff) < 128:
            file = open(P.data_path, 'rb')
            P.block_access += 1
            for key, value in dataDict.items():
                if value == pageId:
                    file.seek(key * 8192)
                    count = 0
                    s = file.read(8192)
                    n = Node()
                    val = dataDict[key]  # 页面号
                    while count < Block.block_size:
                        ss = struct.unpack('dd', s[count * 16:count * 16 + 16])
                        if int(ss[0]) == -1:
                            break
                        p = point()
                        p.pos = ss[0]
                        p.time = ss[1]
                        n.pageList.append(p)
                        count += 1
                    data_buff[val] = n
                    file.close()
                    for i in data_buff[pageId].pageList:
                        if i.pos >= _r.pos_min and i.pos <= _r.pos_max and i.time >= _r.time_min and i.time <= _r.time_max:
                            recall += 1
                    break
        elif len(data_buff) == 128 :  # 缓存已满
            k = list(data_buff.keys())
            del data_buff[k[0]]
            file = open(P.data_path, 'rb')
            P.block_access += 1
            for key, value in dataDict.items():
                if value == pageId:
                    file.seek(key * 8192)
                    count = 0
                    s = file.read(8192)
                    n = Node()
                    val = dataDict[key]  # 页面号
                    while count < Block.block_size:
                        ss = struct.unpack('dd', s[count * 16:count * 16 + 16])
                        if int(ss[0]) == -1:
                            break
                        p = point()
                        p.pos = ss[0]
                        p.time = ss[1]
                        n.pageList.append(p)
                        count += 1
                    data_buff[val] = n
                    file.close()
                    for i in data_buff[pageId].pageList:
                        if i.pos >= _r.pos_min and i.pos <= _r.pos_max and i.time >= _r.time_min and i.time <= _r.time_max:
                            recall += 1
                    break
    return recall


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
        r = region()
        r.pos_min = float(splitData[j][0])
        r.pos_max = float(splitData[j][1])
        r.time_min = float(splitData[j][2])
        r.time_max=float(splitData[j][3])
        range_query.append(r)
    return range_query


def construct_match(pe=1):
    '''
    批量构建
    :param pe: 周期
    '''
    t=KDBTree()
    datasets=[]
    path="..//SIM//EP"+str(pe)
    read_datasets(path, datasets)
    t3=time.time()
    for pt in datasets:
        insert_kdb(t, pt)
    t4 = time.time()
    print("构建时间：",t4-t3)
    sum=0
    for page in t.nodes:
        if page.flag==1:
            sum+=1
    dataIdDict, indexIdDict=write_data(t)
    print("索引块数：", sum, len(indexIdDict))
    print("数据块数：",len(dataIdDict))
    range_query = read_range("..//SIM_DATA//range_sim.txt")  # 读取范围
    search_num=0  # 找到的数据点数
    index_buff={}  # 索引缓冲区
    data_buff={}  #数据缓冲区
    t1=time.time()
    for _r in range_query:
        resultsId=set()
        region_query(t.rootID, _r, resultsId, index_buff, indexIdDict)
        num=range_filter(resultsId, _r, data_buff, dataIdDict)
        search_num+=num
    t2 = time.time()
    print("查询时间：",t2-t1)
    real_num = 0
    for qr in range_query:
        for p in datasets:
            if p.pos >= qr.pos_min and p.pos <= qr.pos_max and p.time >= qr.time_min and p.time <= qr.time_max:
                real_num += 1
    print("找到的数据点数：", search_num)
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：",P.block_access)


def construct_match_real(pe=1):
    '''
    批量构建
    :param pe: 周期
    '''
    t=KDBTree()
    datasets=[]
    path="..//GEO//EP"+str(pe)
    read_datasets(path, datasets)
    t3=time.time()
    for pt in datasets:
        insert_kdb(t, pt)
    t4 = time.time()
    print("构建时间：",t4-t3)
    sum=0
    for page in t.nodes:
        if page.flag==1:
            sum+=1
    P.data_path='..//KDB//geo.dat'
    P.index_path='..//KDB//geo.dix'
    dataIdDict, indexIdDict=write_data(t)
    range_query = read_range("..//GeoData//range_geo.txt")  # 读取范围
    search_num=0  # 找到的数据点数
    index_buff={}  # 索引缓冲区
    data_buff={}  #数据缓冲区
    t1=time.time()
    for _r in range_query:
        resultsId=set()
        region_query(t.rootID, _r, resultsId, index_buff, indexIdDict)
        num=range_filter(resultsId, _r, data_buff, dataIdDict)
        search_num+=num
    t2 = time.time()
    print("查询时间：",t2-t1)
    real_num = 0
    for qr in range_query:
        for p in datasets:
            if p.pos >= qr.pos_min and p.pos <= qr.pos_max and p.time >= qr.time_min and p.time <= qr.time_max:
                real_num += 1
    print("找到的数据点数：", search_num)
    print("实际窗口内的数据点数：", real_num)
    print("访问的磁盘数：",P.block_access)


if __name__ == "__main__":
    #模拟数据
    construct_match(1)
    #真实数据
    # construct_match_real(1)