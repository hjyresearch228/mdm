from sklearn import linear_model
from sklearn.base import *
import time


class CART:
    num_leafNode = 0  # 所有结点数目
    leafNumber = 0  # 叶子结点数目
    max_depth = 1  # 树的高度
    allLeafNode = {}
    DataNum = 0  # 第一期数据数目

    def __init__(self,stop_error=0):
        self.feature = None  # 划分维度
        self.n_samples = None
        self.gain = None  # 最小均方误差
        self.left = None
        self.right = None
        self.threshold = None  # 划分值
        self.depth = 1
        self.model = None
        self.leaf_target = None
        self.leaf_features = None
        self.stop_error = stop_error
        self.Id = None

    def get_liner_MaxError(self, hilbert, target):
        '''
        求得当前线性模型的最大误差和模型
        :param target: 输入
        :param hilbert: 希尔伯特值
        :return: 最大误差,模型,预测值
        '''
        tempModel = linear_model.LinearRegression()
        tempModel.fit(hilbert, target)
        pre = tempModel.predict(hilbert)
        maxError = 0
        for i in range(len(pre)):
            if abs(pre[i] - target[i]) > maxError:
                maxError = abs(pre[i] - target[i])
        return maxError, tempModel,pre

    def fit(self, features, target):
        '''
        开始构建回归模型树
        :param features: 特征(轨迹点集合)
        :param target:输出
        :return:
        '''
        CART.num_leafNode = CART.num_leafNode + 1
        self.Id = CART.num_leafNode
        self.n_samples = features.shape[0]  # 样本点数量
        if self.depth>CART.max_depth:
            CART.max_depth=self.depth
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        minImpurity = float('inf')
        max_error, model,pre = self.get_liner_MaxError(features[:, 2].reshape(-1, 1), target)
        # 如果当前的最大误差已经满足条件，那么此时就停止分类，并且保存
        if max_error < self.stop_error:
            self.feature = None
            self.right = None
            self.left = None
            self.model = model
            self.leaf_features = features
            self.leaf_target = target
            CART.allLeafNode[self.Id] = self
            CART.leafNumber += 1
            return
        else:
            # 找最优分割值
            for col in range(2):  # 每个维度
                feature_level = np.unique(features[:, col])
                thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0
                for threshold in thresholds:  # 分割值
                    target_l = target[features[:, col] <= threshold]
                    features_l = features[features[:, col] <= threshold]
                    max_error_l, model_l, pre_l = self.get_liner_MaxError(features_l[:, 2].reshape(-1, 1), target_l)
                    impurity_l = self._calc_impurity(pre_l,target_l)
                    target_r = target[features[:, col] > threshold]
                    features_r = features[features[:, col] > threshold]
                    max_error_r, model_r, pre_r = self.get_liner_MaxError(features_r[:, 2].reshape(-1, 1), target_r)
                    impurity_r = self._calc_impurity(pre_r,target_r)
                    if max_error_l<self.stop_error and max_error_r<self.stop_error:
                        best_gain = (impurity_l + impurity_r) / self.n_samples
                        best_feature = col
                        best_threshold = threshold
                        self.feature = best_feature  # 划分维度
                        self.gain = best_gain
                        self.threshold = best_threshold  # 划分值
                        # 构造左子树
                        self.left = CART(stop_error=self.stop_error)
                        self.left.depth = self.depth + 1
                        self.left.fit(features_l, target_l)
                        # 构造右子树
                        self.right = CART(stop_error=self.stop_error)
                        self.right.depth = self.depth + 1
                        self.right.fit(features_r, target_r)
                        return
                    if (impurity_l + impurity_r)/self.n_samples < minImpurity:
                        minImpurity = (impurity_l + impurity_r)/self.n_samples
                        best_gain = (impurity_l + impurity_r)/self.n_samples
                        best_feature = col
                        best_threshold = threshold
            self.feature = best_feature  # 划分维度
            self.gain = best_gain
            self.threshold = best_threshold  # 划分值
            #  构造左子树
            features_l = features[features[:, self.feature] <= self.threshold]
            target_l = target[features[:, self.feature] <= self.threshold]
            self.left = CART(stop_error=self.stop_error)
            self.left.depth = self.depth + 1
            self.left.fit(features_l,target_l)
            #  构造右子树
            features_r = features[features[:, self.feature] > self.threshold]
            target_r = target[features[:, self.feature] > self.threshold]
            self.right = CART(stop_error=self.stop_error)
            self.right.depth = self.depth + 1
            self.right.fit(features_r,target_r)

    def _calc_impurity(self, pre,target):
        '''
        计算均方误差
        '''
        sum=0
        for i in range(target.shape[0]):
            sum+=(target[i]-pre[i])**2
        return sum

    def _predict(self,d):
        '''
        预测一个样本点的位置
        :param d: [pos,time,hilbert]
        :return: 预测地址
        '''
        if self.feature != None:  # 非叶子模型
            if d[self.feature] <= self.threshold:
                return self.left._predict(d)
            else:
                return self.right._predict(d)
        else:  # 叶子模型
            pre = self.model.predict(np.array(d[2]).reshape(-1, 1))
            return pre

    def predict(self, d):
        '''
        预测样本集的地址
        :param d:
        :return: 预测值
        '''
        pre=[]
        for i in range(d.shape[0]):
            temp=self._predict(d[i])
            pre.append(temp[0])
        pre=np.array(pre)
        return pre








