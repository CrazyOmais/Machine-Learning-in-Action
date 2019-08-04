import numpy as np
# import matplotlib     这句和下面的from import引用内容似乎是一样的
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import operator

def classify0(inX, dataSet, labels, k):
    '''

    :param inX: 测试集
    :param dataSet: 训练集
    :param labels: 分类标签
    :param k: 选择距离最小的k个点
    :return sortedClassCount[0][0]: 分类结果
    '''
    dataSetSize = dataSet.shape[0]  # numpy函数shape[0]返回dataSet的行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    sqDiffMat = diffMat ** 2    # 二维特征相减后平方
    sqDistances = sqDiffMat.sum(axis = 1)   # sum()所有元素相加,sum(0)列相加,sum(1)行相加
    distances = sqDistances ** 0.5  # 开方,计算出距离
    sortedDistIndices = distances.argsort() # 返回distances中元素从小到大排序后的索引值
    classCount = {} # 定一个记录类别次数的字典

    for i in range(k):
        voteILabel = labels[sortedDistIndices[i]]   # 取出前k个元素的类别
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1  # 计算类别次数
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别

    return sortedClassCount[0][0]

def file2matrix(filename):
    '''
    打开并解析文件，并对数据进行分类，：1 - 不喜欢; 2 - 魅力一般; 3 - 极具魅力
    :param filename: 文件名
    :return returnMat: 特征矩阵
    :return classLabelVector: 分类label向量
    '''
    fr = open(filename)   # 打开文件
    arrayOLines = fr.readlines()    # 读取文件内容
    numberOfLines = len(arrayOLines)    # 得到文件行数
    returnMat = np.zeros((numberOfLines,3)) # 返回值1：特征矩阵，numberOfLines行，3列
    classLabelVector = []   # 返回值2：标签向量
    index = 0   # 行索引值

    for line in arrayOLines:
        line = line.strip() # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        listFromLine = line.split('\t') # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        returnMat[index, :] = listFromLine[0 : 3]   # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵

        # 根据文本中标记的喜欢的程度进行分类
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

def showdatas(datingDataMat, datingLabels):
    '''
    可视化数据
    :param datingDataMat: 特征矩阵
    :param datingLabels: 分类标签
    :return:
    '''
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    # 显示图片
    plt.show()

def autoNorm(dataSet):
    '''
    数据归一化
    :param dataSet: 特征矩阵
    :return normDataSet: 归一化之后的特征矩阵
    :return ranges: 数据范围
    :return minVals: 数据最小值
    '''
    minVals = dataSet.min(0)    # 数据最小值
    maxVals = dataSet.max(0)    # 数据最大值
    ranges = maxVals - minVals  # 数据范围
    normDataSet = np.zeros(np.shape(dataSet))

    m = dataSet.shape[0]    # 返回dataSet的行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))    # 原始值减去最小值，使得数据下限为0
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   # 除以最大值和最小值的差，使得数据上限为1

    return normDataSet, ranges, minVals

def datingClassTest():
    '''
    kNN的分类器测试函数
    :return normDataSet: 归一化之后的特征矩阵
    :return ranges: 数据范围
    :return minVals: 数据最小值
    '''
    filename = 'datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename)

    hoRatio = 0.10  # 取所有数据的百分之十进行测试
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化处理
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 测试集数据个数
    errorCount = 0.0    # 分类错误计数

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('错误率：%f%%' %(errorCount / float(numTestVecs) * 100))



if __name__ == '__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print('原始数据集：')
    print(datingDataMat)
    print('标签向量：')
    print(datingLabels)
    print('归一化之后的数据集：')
    print(normDataSet)
    print('数据范围：', end = ' ')
    print(ranges)
    print('数据最小值：', end = ' ')
    print(minVals)

    showdatas(normDataSet, datingLabels)
    datingClassTest()

