# "Learned Index for Spatial Queries"版本, 可以用
def z_order_mapping(dim, order, point):
    """
    将点坐标映射为Z曲线值
    :param dim: 维度
    :param order: 阶数
    :param point: 点坐标
    :return: Z曲线值
    """
    bi_num = pow(2, order)                      # 二进制位数
    bi_zv = ""                                  # Z曲线值二进制字符串
    bi_point = [0] * dim                        # 二进制点坐标
    # 转化为二进制字符串
    for i in range(dim):
        bi_point[i] = bin(point[i])[2:].zfill(bi_num)
    # 计算Z曲线值二进制字符串
    for j in range(bi_num):
        for i in range(dim):
            bi_zv += bi_point[i][j:j + 1]
    # 转化为Z曲线值
    z_value = int(bi_zv, 2)
    return z_value