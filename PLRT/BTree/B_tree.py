from random import randint, randrange
from bisect import  bisect_left

class Node:
    def __init__(self):
        self.leaf = True
        self.keys = []
        self.children = []
        self.parent=None
        self.pointer=None

class BTree:
    leaf_list=[]  # 叶子结点
    node_num=0  # 内结点个数
    pointer_num=0  # 指针个数

    def __init__(self, t):
        """
        Initializing the B-Tree
        :param t: Order.
        """
        self.root = Node()
        self.t = t

    def printTree(self, node, lvl=0):
        print("Level ", lvl, " --> ", len(node.keys), end=": ")
        for i in node.keys:
            print(i, end=" ")
        print()
        lvl += 1
        if len(node.children) > 0:
            BTree.node_num += 1
            BTree.pointer_num+=len(node.children)
            for i in node.children:
                self.printTree(i, lvl)
        else:
            BTree.pointer_num = BTree.pointer_num+len(node.keys)+1
            BTree.leaf_list.append(node)


    def join(self):
        for i in range(len(BTree.leaf_list)-1):
            BTree.leaf_list[i].pointer=BTree.leaf_list[i+1]

    def search(self, node, k):
        if not node:
            return
        if node.leaf:
            return node
        else:
            ind = bisect_left(node.keys, k)
            return self.search(node.children[ind],k)


    def insert(self,node,k):
        """
        Calls the respective helper functions for insertion into B-Tree
        :param k: The key to be inserted.
        """
        if node.leaf:
            if len(node.keys)<self.t-1:
                node.keys.append(k)
                node.keys.sort()
            else:
                node.keys.append(k)
                node.keys.sort()
                self.split(node)
        else:
            ind=bisect_left(node.keys,k)
            self.insert(node.children[ind],k)


    def split(self, node):
        mid=len(node.keys)//2
        left=Node()
        left.keys.extend(node.keys[:mid])
        right = Node()
        right.keys.extend(node.keys[mid+1:])
        if not node.leaf:
            left.leaf=False
            right.leaf=False
        if len(node.children)>0:
            left.children.extend(node.children[:mid+1])
            right.children.extend(node.children[mid+1:])
        if not node.parent:
            new_root=Node()
            new_root.leaf=False
            new_root.keys.append(node.keys[mid])
            self.root=new_root
            left.parent=new_root
            right.parent=new_root
            new_root.children.append(left)
            new_root.children.append(right)
        else:
            node.parent.keys.append(node.keys[mid])
            node.parent.keys.sort()
            ind=node.parent.children.index(node)
            left.parent = node.parent
            right.parent = node.parent
            node.parent.children[ind]=left
            if ind==len(node.parent.children)-1:
                node.parent.children.append(right)
            else:
                node.parent.children.insert(ind+1,right)
            if len(node.parent.keys)>self.t-1:
                self.split(node.parent)


# Program starts here
if __name__ == '__main__':
    B = BTree(5)
    # Insert
    for i in range(1, 18, 2):
        B.insert(B.root, i)
    B.printTree(B.root)  # 打印树
    print(B.search(B.root, 15))
