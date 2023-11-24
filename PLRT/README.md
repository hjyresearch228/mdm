# 数据集和源代码说明

## 一、数据集

1. SIM目录下放置模拟数据的500个范围查询。

2. GEO目录下放置真实数据的500个范围查询。

## 二、源代码结构

1. GEO/regionForRealFor5wOne.txt：真实数据上的500个范围查询。

2. SIM /regionForSim.txt：模拟数据上的500个范围查询。

3. BTree/B_tree.py：B+树模型

4. linear_model_57.py：用于训练和测试分段线性回归树

## 三、主程序入口

整个程序己经被完全封装，只需要运行linear_model_57.py文件就可以完成论文中提到的所有试验。

linear_model_57.py文件中调用的函数描述：

1. sim_test：在模拟数据集上构造分段线性回归树，并进行查询操作。

2. geo_test：在真实数据集上构造分段线性回归树，并进行查询操作。
