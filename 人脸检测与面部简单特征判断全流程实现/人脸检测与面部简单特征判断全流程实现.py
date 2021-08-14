#!/usr/bin/env python
# coding: utf-8

# ## 一 项目简介
# > 解决PaddleDetection版本问题导致部署时模型加载问题，此版本为PaddleDetection回溯版本，PaddleDetection同步在数据集中
# - 1 该项目使用Labelimg进行数据标注，自定义数据集；
# - 2 使用Paddlex将数据集划分为训练集、测试集；
# - 3 使用PaddleDetection目标检测套件训练模型；
# - 4 最后导出模型，通过PaddleLite生成.nb文件，部署到手机上；
# - 5 安卓部署详细操作；
# - （1）配置系统环境
# - （2）下载PaddleLite-dome
# - （3）下载Android Studio开发环境并配置软件环境
# - （4）dome调试
# - （5）根据dome，加入自己的模型，修改配置，实现自己的dome，并调试
# - （6）将APP打包成可安装程序.apk文件
# - 实现飞桨框架深度学习模型从0到1的全流程。

# ## 二 数据标注
# - 安装Anaconda便于对包环境的管理，在Anaconda环境中安装Labelimg（安装教程链接：https://blog.csdn.net/MrSong007/article/details/93641670）
# - 下面讲解安装成功后操作流程：参考链接：https://cloud.tencent.com/developer/news/325876#:~:text=%20%20LabelImg%E6%98%AF%E7%94%A8%E4%BA%8E%E5%88%B6%E4%BD%9CVOC%E6%95%B0%E6%8D%AE%E9%9B%86%E6%97%B6%EF%BC%8C%E5%AF%B9%E6%95%B0%E6%8D%AE%E9%9B%86%E8%BF%9B%E8%A1%8C%E6%A0%87%E6%B3%A8%E7%9A%84%E5%B7%A5%E5%85%B7%E3%80%82,%E7%B3%BB%E7%BB%9F%EF%BC%9Awin10%20%E8%BD%AF%E4%BB%B6%EF%BC%9Aanaconda3%201.%E5%AE%89%E8%A3%85anaconda3
# - 1 新建数据集文件夹：JPEGImages文件存放事先准备好的图片，Annotations文件存放xml标注文件（未标注时此文件为空）
# - 2 打开Labelimg：点击Change Save Dir找到刚刚创建的Annotations文件；点击Open Dir找到JPEGImages文件；快捷键按D，拖拽选中区域，并在弹框内打标签；点击Next Image对下一张图片进行标注（此时会弹出是否保存的提示框，可勾选View->Auto Save mode，默认将每张图片标注完后自动保存）
# - 3 上述步骤完成后，Annotations文件中会产生一堆xml文件，格式如下，
# - 4 最后将文件压缩上传到aistudio

# ## 三 paddlex划分数据集

# In[1]:


#将数据集进行解压（此处可换成你自己标注的数据集）
get_ipython().system('unzip data/data104195/dataxiao.zip -d masks')


# In[2]:


# 准备一些可能用到的工具库
import xml.etree.cElementTree as ET
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import paddle.fluid as fluid
import time
import random


# In[3]:


# 导入paddlex
get_ipython().system('pip install paddlex')


# In[4]:


# 使用paddlex进行数据集的划分训练集、验证集、测试集
get_ipython().system('paddlex --split_dataset --format VOC --dataset_dir masks/VOC_MASK --val_value 0.2 --test_value 0.1')


# 执行完上述代码可以看到，masks/VOC_MASK目录下会自动生成以下文件：
# 
# - val_list.txt 验证集列表
# - test_list.txt 测试集列表
# - labels.txt 标签列表
# - train_list.txt 训练集列表

# ## 四 PaddleDetection目标检测套件使用，进行模型的创建、训练、导出

# ### 1 PaddleDetection下载、配置、数据集文件的处理

# In[5]:


# 下载PaddleDetection到本地
# 为避免因PaddleDetection更新导致代码报错，我将我的PaddleDetection版本挂载到数据集里了data/data88824/model_PaddleDetection.zip，可解压使用
# !git clone https://gitee.com/paddlepaddle/PaddleDetection.git
get_ipython().system('unzip data/data104195/model_PaddleDetection.zip')
get_ipython().system('unzip PaddleDetection.zip -d PaddleDetection')


# In[6]:


# 把PaddleDetection转移到work目录下，可以持久保存
get_ipython().system('mv PaddleDetection work/')


# In[7]:


#配置PaddleDetection环境
get_ipython().system('pip install -r work/PaddleDetection/requirements.txt')


# In[8]:


#将数据集转到PaddleDetection/dataset，专门存放和处理数据集的文件
get_ipython().system('cp -r masks/VOC_MASK work/PaddleDetection/dataset/')


# In[9]:


get_ipython().run_line_magic('cd', 'work/PaddleDetection')


# In[10]:


# 调整下标注文件命名，与PaddleDetection默认的一致（处理成相同文件名）
get_ipython().system('mv dataset/VOC_MASK/labels.txt dataset/VOC_MASK/label_list.txt')


# ### 2 修改模型文件

# #### 修改模型文件ssd_mobilenet_v1_voc.yml
# 需对配置文件进行修改，num_classes: 5
# 
# 此处用的SSD模型，主干网络是Mobilnet（比VGG更轻量级的网络，适合于移动端部署），图片输入尺寸300x300，数据集格式voc
# 
# 地址：work/PaddleDetection/configs/ssd/ssd_mobilenet_v1_voc.yml
# 
# ```
# architecture: SSD
# pretrain_weights: https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_coco_pretrained.tar
# use_gpu: true
# # max_iters计算方法：训练集文件数*训练epoch数（这里按30轮计算）/训练集batchsize
# max_iters: 5219
# snapshot_iter: 2000
# log_iter: 100
# metric: VOC
# map_type: 11point
# save_dir: output
# weights: output/ssd_mobilenet_v1_voc/model_final
# # 2(label_class) + 1(background) 在口罩数据集中，具体类别为：人脸、戴口罩人脸、背景类
# num_classes: 5
# 
# SSD:
#   backbone: MobileNet
#   multi_box_head: MultiBoxHead
#   output_decoder:
#     background_label: 0
#     keep_top_k: 200
#     nms_eta: 1.0
#     nms_threshold: 0.45
#     nms_top_k: 400
#     score_threshold: 0.01
# 
# MobileNet:
#   norm_decay: 0.
#   conv_group_scale: 1
#   conv_learning_rate: 0.1
#   extra_block_filters: [[256, 512], [128, 256], [128, 256], [64, 128]]
#   with_extra_blocks: true
# 
# MultiBoxHead:
#   aspect_ratios: [[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]]
#   base_size: 300
#   flip: true
#   max_ratio: 90
#   max_sizes: [[], 150.0, 195.0, 240.0, 285.0, 300.0]
#   min_ratio: 20
#   min_sizes: [60.0, 105.0, 150.0, 195.0, 240.0, 285.0]
#   offset: 0.5
# 
# LearningRate:
#   schedulers:
#   - !PiecewiseDecay
#     milestones: [10000, 15000, 20000, 25000]
#     values: [0.001, 0.0005, 0.00025, 0.0001, 0.00001]
# 
# OptimizerBuilder:
#   optimizer:
#     momentum: 0.0
#     type: RMSPropOptimizer
#   regularizer:
#     factor: 0.00005
#     type: L2
# 
# TrainReader:
#   inputs_def:
#     image_shape: [3, 300, 300]
#     fields: ['image', 'gt_bbox', 'gt_class']
#   dataset:
#     !VOCDataSet
#     anno_path: train_list.txt
#     dataset_dir: dataset/VOC_MASK
#     use_default_label: false
#   sample_transforms:
#   - !DecodeImage
#     to_rgb: true
#   - !RandomDistort
#     brightness_lower: 0.875
#     brightness_upper: 1.125
#     is_order: true
#   - !RandomExpand
#     fill_value: [127.5, 127.5, 127.5]
#   - !RandomCrop
#     allow_no_crop: false
#   - !NormalizeBox {}
#   - !ResizeImage
#     interp: 1
#     target_size: 300
#     use_cv2: false
#   - !RandomFlipImage
#     is_normalized: true
#   - !Permute {}
#   - !NormalizeImage
#     is_scale: false
#     mean: [127.5, 127.5, 127.5]
#     std: [127.502231, 127.502231, 127.502231]
#   batch_size: 32
#   shuffle: true
#   drop_last: true
#   worker_num: 8
#   bufsize: 16
#   use_process: true
# 
# EvalReader:
#   inputs_def:
#     image_shape: [3, 300, 300]
#     fields: ['image', 'gt_bbox', 'gt_class', 'im_shape', 'im_id', 'is_difficult']
#   dataset:
#     !VOCDataSet
#     anno_path: val_list.txt
#     dataset_dir: dataset/VOC_MASK
#     use_default_label: false
#   sample_transforms:
#   - !DecodeImage
#     to_rgb: true
#   - !NormalizeBox {}
#   - !ResizeImage
#     interp: 1
#     target_size: 300
#     use_cv2: false
#   - !Permute {}
#   - !NormalizeImage
#     is_scale: false
#     mean: [127.5, 127.5, 127.5]
#     std: [127.502231, 127.502231, 127.502231]
#   batch_size: 32
#   worker_num: 8
#   bufsize: 16
#   use_process: false
# 
# TestReader:
#   inputs_def:
#     image_shape: [3,300,300]
#     fields: ['image', 'im_id', 'im_shape']
#   dataset:
#     !ImageFolder
#     anno_path: dataset/VOC_MASK/label_list.txt
#     use_default_label: false
#   sample_transforms:
#   - !DecodeImage
#     to_rgb: true
#   - !ResizeImage
#     interp: 1
#     max_size: 0
#     target_size: 300
#     use_cv2: true
#   - !Permute {}
#   - !NormalizeImage
#     is_scale: false
#     mean: [127.5, 127.5, 127.5]
#     std: [127.502231, 127.502231, 127.502231]
#   batch_size: 1
# ```

# ### 3 模型训练

# In[13]:


# 配置文件在work/PaddleDetection/configs/ssd/ssd_mobilenet_v1_voc.yml，模型保存在work/PaddleDetection/output
# 可使用可视化工具VisualDL,文件在work/PaddleDetection/vdl_log_dir（需要模型运行时才能产出）
# !python -u tools/train.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --use_vdl True --eval
get_ipython().system('python -u tools/train.py -c configs/ssd/ssd_mobilenet_v1_voc.yml --use_vdl True --eval')


# ### 4 模型预测效果展示

# 第一张测试图片

# In[5]:


#测试，查看模型效果，结果存放在work/PaddleDetection/output
get_ipython().run_line_magic('cd', '/home/aistudio/work/PaddleDetection/')
# !python tools/infer.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --infer_img=/home/aistudio/001.jpg -o weights=output/ssd_mobilenet_v1_300_120e_voc/best_model.pdparams
get_ipython().system('python tools/infer.py -c configs/ssd/ssd_mobilenet_v1_voc.yml --infer_img=/home/aistudio/work/example/001.jpg -o weights=output/ssd_mobilenet_v1_voc/best_model.pdparams')


# In[6]:


get_ipython().run_line_magic('cd', '/home/aistudio/')


# In[7]:


import matplotlib.pyplot as plt
import PIL.Image as Image


# In[8]:


path='work/PaddleDetection/output/001.jpg'
img = Image.open(path)
plt.imshow(img)          #根据数组绘制图像
plt.show()               #显示图像


# 第二张测试图片

# In[9]:


#测试，查看模型效果，结果存放在work/PaddleDetection/output
get_ipython().run_line_magic('cd', '/home/aistudio/work/PaddleDetection/')
# !python tools/infer.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --infer_img=/home/aistudio/001.jpg -o weights=output/ssd_mobilenet_v1_300_120e_voc/best_model.pdparams
get_ipython().system('python tools/infer.py -c configs/ssd/ssd_mobilenet_v1_voc.yml --infer_img=/home/aistudio/work/example/002.jpg -o weights=output/ssd_mobilenet_v1_voc/best_model.pdparams')


# In[10]:


get_ipython().run_line_magic('cd', '/home/aistudio/')


# In[11]:


import matplotlib.pyplot as plt
import PIL.Image as Image
path='work/PaddleDetection/output/002.jpg'
img = Image.open(path)
plt.imshow(img)          #根据数组绘制图像
plt.show()               #显示图像


# 第3张图片

# In[12]:


#测试，查看模型效果，结果存放在work/PaddleDetection/output
get_ipython().run_line_magic('cd', '/home/aistudio/work/PaddleDetection/')
# !python tools/infer.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --infer_img=/home/aistudio/001.jpg -o weights=output/ssd_mobilenet_v1_300_120e_voc/best_model.pdparams
get_ipython().system('python tools/infer.py -c configs/ssd/ssd_mobilenet_v1_voc.yml --infer_img=/home/aistudio/work/example/003.jpg -o weights=output/ssd_mobilenet_v1_voc/best_model.pdparams')


# In[13]:


get_ipython().run_line_magic('cd', '/home/aistudio/')


# In[14]:


import matplotlib.pyplot as plt
import PIL.Image as Image
path='work/PaddleDetection/output/003.jpg'
img = Image.open(path)
plt.imshow(img)          #根据数组绘制图像
plt.show()               #显示图像


# ### 5 模型导出

# In[24]:


get_ipython().run_line_magic('cd', '/home/aistudio/work/PaddleDetection/')


# In[25]:


# 导出模型
#!python tools/export_model.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml -o weights=output/ssd_mobilenet_v1_300_120e_voc/best_model.pdparams --output_dir ./inference
get_ipython().system('python tools/export_model.py -c configs/ssd/ssd_mobilenet_v1_voc.yml -o weights=output/ssd_mobilenet_v1_voc/model_final.pdparams --output_dir ./inference')


# In[ ]:


# 导出YOLOv3模型
python tools/export_model.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o weights=output/yolov3_mobilenet_v1_roadsign/best_model.pdparams


# ## 五 PaddleLite生成.nb模型文件
# [PaddleLite文档](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/android_app_demo.html)

# In[26]:


# 准备PaddleLite依赖
get_ipython().system('pip install paddlelite==2.9.0')


# In[ ]:


get_ipython().run_line_magic('cd', '/home/aistudio/work/PaddleDetection/')


# In[27]:


# 准备PaddleLite部署模型
#--valid_targets中参数（arm）用于传统手机，（npu,arm ）用于华为带有npu处理器的手机
get_ipython().system('paddle_lite_opt     --model_file=inference/ssd_mobilenet_v1_voc/__model__     --param_file=inference/ssd_mobilenet_v1_voc/__params__     --optimize_out=./inference/mask_ssd_mobilenet_v1_voc     --optimize_out_type=naive_buffer     --valid_targets=arm ')
    #--valid_targets=npu,arm 


# ## 六 安卓端部署

# ### 使用EasyEdge部署模型
# - 选择本地模型上传
# - 选择物体检测模块
# - 选择SSD，MobileNetV1-SSD模型
# - 分别上传模型文件__model__，参数文件__params__，标签文件，前两个文件在./inference/ssd_mobilenet_v1_voc/文件夹下
# - 填写其余描述后，各个文件校验合格后即可点击确定。
# - 选择部署环境，生成SDK即可，之后可扫码下载apk文件。
