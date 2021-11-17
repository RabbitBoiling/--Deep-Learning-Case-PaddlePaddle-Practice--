import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

minmax_scale = MinMaxScaler(feature_range=(0,1))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/power decreasing"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_power_decreasing = data_txts
data_power_decreasing_BN = minmax_scale.fit_transform(data_power_decreasing)
power_decreasing_one_hot = np.full((data_power_decreasing.shape[0],10),0)
power_decreasing_one_hot[:,0] = 1.
Data_power_decreasing = np.hstack((data_power_decreasing_BN,power_decreasing_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/PRZ liquid space leak"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_PRZ_liquid = data_txts
data_PRZ_liquid_BN = minmax_scale.fit_transform(data_PRZ_liquid)
PRZ_liquid_one_hot = np.full((data_PRZ_liquid.shape[0],10),0)
PRZ_liquid_one_hot[:,1] = 1.
Data_PRZ_liquid = np.hstack((data_PRZ_liquid_BN,PRZ_liquid_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/PRZ vaper space leak"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_PRZ_vaper = data_txts
data_PRZ_vaper_BN = minmax_scale.fit_transform(data_PRZ_vaper)
PRZ_vaper_one_hot = np.full((data_PRZ_vaper.shape[0],10),0)
PRZ_vaper_one_hot[:,2] = 1.
Data_PRZ_vaper = np.hstack((data_PRZ_vaper_BN,PRZ_vaper_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/RCS CL LOCA 1"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_RCS_CL_1 = data_txts
data_RCS_CL_1_BN = minmax_scale.fit_transform(data_RCS_CL_1)
RCS_CL_1_one_hot = np.full((data_RCS_CL_1.shape[0],10),0)
RCS_CL_1_one_hot[:,3] = 1.
Data_RCS_CL_1 = np.hstack((data_RCS_CL_1_BN,RCS_CL_1_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/RCS CL LOCA 2"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_RCS_CL_2 = data_txts
data_RCS_CL_2_BN = minmax_scale.fit_transform(data_RCS_CL_2)
RCS_CL_2_one_hot = np.full((data_RCS_CL_2.shape[0],10),0)
RCS_CL_2_one_hot[:,4] = 1.
Data_RCS_CL_2 = np.hstack((data_RCS_CL_2_BN,RCS_CL_2_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/RCS HL LOCA 1"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_RCS_HL_1 = data_txts
data_RCS_HL_1_BN = minmax_scale.fit_transform(data_RCS_HL_1)
RCS_HL_1_one_hot = np.full((data_RCS_HL_1.shape[0],10),0)
RCS_HL_1_one_hot[:,5] = 1.
Data_RCS_HL_1 = np.hstack((data_RCS_HL_1_BN,RCS_HL_1_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/RCS HL LOCA 2"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_RCS_HL_2 = data_txts
data_RCS_HL_2_BN = minmax_scale.fit_transform(data_RCS_HL_2)
RCS_HL_2_one_hot = np.full((data_RCS_HL_2.shape[0],10),0)
RCS_HL_2_one_hot[:,6] = 1.
Data_RCS_HL_2 = np.hstack((data_RCS_HL_2_BN,RCS_HL_2_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/SG 2nd side leak"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_SG_2nd = data_txts
data_SG_2nd_BN = minmax_scale.fit_transform(data_SG_2nd)
SG_2nd_one_hot = np.full((data_SG_2nd.shape[0],10),0)
SG_2nd_one_hot[:,7] = 1.
Data_SG_2nd = np.hstack((data_SG_2nd_BN,SG_2nd_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/SGTR60功率"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_SGTR60 = data_txts
data_SGTR60_BN = minmax_scale.fit_transform(data_SGTR60)
SGTR60_one_hot = np.full((data_SGTR60.shape[0],10),0)
SGTR60_one_hot[:,8] = 1.
Data_SGTR60 = np.hstack((data_SGTR60_BN,SGTR60_one_hot))

# 文件夹目录
path = "C:/Users/Biao/Desktop/Original Data/SGTR满功率"
# 得到文件夹下的所有文件名称
files= os.listdir(path)
data_txts = np.zeros((1,29),dtype = np.float64)
for file in files:
    # 构造绝对路径
    position = path+'/'+ file
    # print(position)
    # 打开文件
    data_txt = np.loadtxt(position, delimiter=',')
    data_txts = np.append(data_txts,data_txt,axis=0)
data_txts = data_txts[1:,:]
data_SGTR100 = data_txts
data_SGTR100_BN = minmax_scale.fit_transform(data_SGTR100)
SGTR100_one_hot = np.full((data_SGTR100.shape[0],10),0)
SGTR100_one_hot[:,9] = 1.
Data_SGTR100 = np.hstack((data_SGTR100_BN,SGTR100_one_hot))

Data = np.concatenate((Data_power_decreasing,Data_PRZ_liquid,Data_PRZ_vaper,Data_RCS_CL_1,Data_RCS_CL_2,Data_RCS_HL_1,Data_RCS_HL_2,Data_SG_2nd,Data_SGTR60,Data_SGTR100), axis=0)

np.random.shuffle(Data)

N = int(0.8 * Data.shape[0])
Data_train = Data[:N,:]
Data_test = Data[N:,:]



