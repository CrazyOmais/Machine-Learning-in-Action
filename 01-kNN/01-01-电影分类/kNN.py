from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    :param inX: 用于分类的输入向量(测试集)
    :param dataSet: 输入的训练样本集(训练集)
    :param labels: 标签向量
    :param k: 用于选择最近邻的数目
    :return sortedClassCount[0][0]: 分类结果
    '''
    dataSetSize = dataSet.shape[0]  # numpy函数shape[0]返回dataSet的行数

    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
                                                    # numpy函数tile() 把一个数组a,当做模板,重复几次,生成另一个数组b
    sqDiffMat = diffMat ** 2    # 二维特征相减后平方
    sqDistances = sqDiffMat.sum(axis = 1)    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    distances = sqDistances ** 0.5   # 开方，计算出距离
    sortedDistIndicies = distances.argsort()    # 返回distances中元素从小到大排序后的索引值
    classCount = {}     # 定义一个记录类别次数的字典

    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 取出前k个元素的类别
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #计算类别次数
                        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。

    # 排序
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True)
                        # python3中用items()替换python2中的iteritems()
                        # key = operator.itemgetter(1)根据字典的值进行排序
                        # key = operator.itemgetter(0)根据字典的键进行排序
                        # reverse降序排序字典

    return sortedClassCount[0][0]   # 返回次数最多的类别,即所要分类的类别


