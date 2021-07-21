import cv2
import random
import numpy as np
import os
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, AdaptiveAvgPool2D, Linear
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文（黑体）标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1)) # transpose作用是改变序列，(2,0,1)表示各个轴
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

# 定义训练集数据读取器
def train_data_loader(datadir, batch_size=10, mode = 'train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            if name[4] != '7':
                if name[4] != '9':
                    filepath = os.path.join(datadir, name)
                    img = cv2.imread(filepath) # 默认读入一副彩色图片,flags参数可以调
                    img = transform_img(img)
                    if name[0] == 'c':
                        # c开头的图片是猫
                        label = 0
                    elif name[0] == 'd':
                        # d开头的图片是狗
                        label = 1
                    else:
                        raise('Not excepted file name')
                    # 每读取一个样本的数据，就将其放入数据列表中
                    batch_imgs.append(img)
                    batch_labels.append(label)
                    if len(batch_imgs) == batch_size:
                        # 当数据列表的长度等于batch_size的时候，
                        # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                        imgs_array = np.array(batch_imgs).astype('float32') # imgs_array的形状是[batch_size,3,224,224]
                        labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1) # 把labels_array的形状由一维的[batch_size,]变为二维的[batch_size,1]
                        yield imgs_array, labels_array
                        batch_imgs = []
                        batch_labels = []

                if len(batch_imgs) > 0:
                    # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
                    imgs_array = np.array(batch_imgs).astype('float32')
                    labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                    yield imgs_array, labels_array
    return reader

# 定义验证集数据读取器
def valid_data_loader(datadir, batch_size=10, mode='eval'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'eval':
            # 验证集随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            if name[4] == '7' or name[4] == '9':
                filepath = os.path.join(datadir, name)
                img = cv2.imread(filepath)  # 默认读入一副彩色图片,flags参数可以调
                img = transform_img(img)
                if name[0] == 'c':
                    # c开头的图片是猫
                    label = 0
                elif name[0] == 'd':
                    # d开头的图片是狗
                    label = 1
                else:
                    raise ('Not excepted file name')
                # 每读取一个样本的数据，就将其放入数据列表中
                batch_imgs.append(img)
                batch_labels.append(label)
                if len(batch_imgs) == batch_size:
                    # 当数据列表的长度等于batch_size的时候，
                    # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                    imgs_array = np.array(batch_imgs).astype('float32')  # imgs_array的形状是[batch_size,3,224,224]
                    labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)  # 把labels_array的形状由一维的[batch_size,]变为二维的[batch_size,1]
                    yield imgs_array, labels_array
                    batch_imgs = []
                    batch_labels = []

            if len(batch_imgs) > 0:
                # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
    return reader

# 定义测试集数据读取器
def test_data_loader(datadir, batch_size=10, mode='test'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        batch_imgs = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)  # 默认读入一副彩色图片,flags参数可以调
            img = transform_img(img)
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')  # imgs_array的形状是[batch_size,3,224,224]
                yield imgs_array
                batch_imgs = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            yield imgs_array
    return reader

DATADIR = 'D:/Pycharm 2020/Data set/Cat and dog/'
DATADIR_2 = 'C:/Users/Biao/Desktop/Test/'
# 定义Inception块
class Inception(paddle.nn.Layer):
    def __init__(self, c0, c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码，

        c1,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2,图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list,
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3,图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list,
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        '''
        super(Inception, self).__init__()
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(in_channels=c0, out_channels=c1, kernel_size=1, stride=1)
        self.p2_1 = Conv2D(in_channels=c0, out_channels=c2[0], kernel_size=1, stride=1)
        self.p2_2 = Conv2D(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1, stride=1)
        self.p3_1 = Conv2D(in_channels=c0, out_channels=c3[0], kernel_size=1, stride=1)
        self.p3_2 = Conv2D(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2, stride=1)
        self.p4_1 = MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2D(in_channels=c0, out_channels=c4, kernel_size=1, stride=1)

        # # 新加一层batchnorm稳定收敛
        # self.batchnorm = paddle.nn.BatchNorm2D(c1+c2[1]+c3[1]+c4)

    def forward(self, x):
        # 支路1只包含一个1x1卷积
        p1 = F.relu(self.p1_1(x))
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # 支路4包含 最大池化和1x1卷积
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return paddle.concat([p1, p2, p3, p4], axis=1)
        # return self.batchnorm()


class GoogLeNet(paddle.nn.Layer):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # GoogLeNet包含五个模块，每个模块后面紧跟一个池化层
        # 第一个模块包含1个卷积层
        self.conv1 = Conv2D(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=1)
        # 3x3最大池化
        self.pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第二个模块包含2个卷积层
        self.conv2_1 = Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1)
        # 3x3最大池化
        self.pool2 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        # 3x3最大池化
        self.pool3 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        # 3x3最大池化
        self.pool4 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        # 全局池化，用的是global_pooling，不需要设置pool_stride
        self.pool5 = AdaptiveAvgPool2D(output_size=1)
        self.fc = Linear(in_features=1024, out_features=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.pool3(self.block3_2(self.block3_1(x)))
        x = self.block4_3(self.block4_2(self.block4_1(x)))
        x = self.pool4(self.block4_5(self.block4_4(x)))
        x = self.pool5(self.block5_2(self.block5_1(x)))
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


# 定义训练过程
def train_pm(model, optimizer):
    # 开启0号GPU训练
    paddle.set_device('gpu:0')
    print('start training ... ')
    model.train()
    epoch_num = 2
    epoch = 0
    epochs = []
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    train_losses_set = []
    valid_losses_set = []
    train_accuracies_set = []
    valid_accuracies_set = []
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = train_data_loader(DATADIR, batch_size=10, mode='train')
    valid_loader = valid_data_loader(DATADIR, batch_size=10, mode='eval')
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()): # batch_id从1开始一直计数到训练集总个数
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(img) # 形状从shape=[1, 1]开始一直到shape=[batch_size, 1]
            loss = F.binary_cross_entropy_with_logits(logits, label) # loss的shape=[1]，是一个数
            # 每一次for都会产生一个新的loss，append之后个数从1增加至训练集总个数
            # avg_loss = paddle.mean(loss) # 默认值为None，沿着所有轴计算平均值
            train_losses.append(loss.numpy()) # 是一个列表
            pred = F.sigmoid(logits)
            # 计算预测概率小于0.5的类别
            pred2 = pred * (-1.0) + 1.0
            # 得到两个类别的预测概率，并沿第一个维度级联
            pred = paddle.concat([pred2, pred], axis=1)  # 行拼接
            # 两个二维数组做计算
            acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64'))  # paddle.cast转换数据类型
            train_accuracies.append(acc.numpy())
            if batch_id % 20 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, accuracy is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            # 反向传播，更新权重，清除梯度
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # 求平均精度
        train_losses_mean = np.array(train_losses).mean() # 列表转化为numpy数组，不指定axis，是所有维度求均值
        train_losses_set.append(train_losses_mean)
        train_accuracies_mean = np.array(train_accuracies).mean()
        train_accuracies_set.append(train_accuracies_mean)
        epoch = epoch + 1
        epochs.append(epoch)
        print("[train] accuracy/loss: {}/{}".format(np.mean(train_accuracies), np.mean(train_losses)))

        model.eval()
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(img)
            # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
            # 计算sigmoid后的预测概率，进行loss计算
            loss = F.binary_cross_entropy_with_logits(logits, label)
            pred = F.sigmoid(logits)
            # 计算预测概率小于0.5的类别
            pred2 = pred * (-1.0) + 1.0
            # 得到两个类别的预测概率，并沿第一个维度级联
            pred = paddle.concat([pred2, pred], axis=1) # 行拼接
            # 两个二维数组做计算
            acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64')) # paddle.cast转换数据类型

            valid_accuracies.append(acc.numpy())
            valid_losses.append(loss.numpy())

        # 求平均精度与损失值
        valid_losses_mean = np.array(valid_losses).mean()
        valid_losses_set.append(valid_losses_mean)
        valid_accuracies_mean = np.array(valid_accuracies).mean()
        valid_accuracies_set.append(valid_accuracies_mean)
        # np.mean()函数里的参数如果不是ndarray类型，将会自动进行转换
        # https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        print("[validation] accuracy/loss: {}/{}".format(np.mean(valid_accuracies), np.mean(valid_losses)))
        model.train()

    # 保存模型
    paddle.save(model.state_dict(), 'palm.pdparams')
    print("模型保存成功，模型参数保存在LR_model.pdparams中")
    paddle.save(optimizer.state_dict(), 'palm.pdopt')

    return epochs, train_accuracies_set, train_losses_set, valid_accuracies_set, valid_losses_set


# 定义测试过程
def test_pm(model):
    # 开启0号GPU训练
    paddle.set_device('gpu:0')
    print('start evaluation .......')

    # 加载模型参数
    params_file_path = './palm.pdparams'
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)

    model.eval()
    eval_loader = test_data_loader(DATADIR_2, batch_size=10, mode='eval')

    for batch_id, data in enumerate(eval_loader()):
        x_data = data
        img = paddle.to_tensor(x_data)
        # 计算预测和精度
        prediction = model(img)
        # print(prediction)
        pred_CatDog = F.sigmoid(prediction)
        pred_CatDog_numpy = np.array(pred_CatDog)
    for i in range(0, pred_CatDog_numpy.shape[0]):
        if pred_CatDog_numpy[i,0] <=0.5:
            print('模型预测的概率为{}，判定该涨图片为（猫）'.format(pred_CatDog_numpy[i,0]))
        else:
            print('模型预测的概率为{}，判定该涨图片为（狗）'.format(pred_CatDog_numpy[i,0]))


# 创建模型
model = GoogLeNet()
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
# 启动训练过程
Epochs, Train_accuracies_set, Train_losses_set, Valid_accuracies_set, Valid_losses_set = train_pm(model, optimizer=opt)

# 画出训练过程中Loss的变化曲线
plt.figure()
plt.title("训练过程的准确率与损失值", fontsize=24)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("accuracy and loss", fontsize=14)
plt.plot(Epochs, Train_accuracies_set, color='red', label='Train accuracies')
plt.plot(Epochs, Train_losses_set, color='yellow', label='Train losses')
plt.plot(Epochs, Valid_accuracies_set, color='green', label='Valid accuracies')
plt.plot(Epochs, Valid_losses_set, color='blue', label='Valid losses')
plt.grid()
plt.show()

# 开启测试
model = GoogLeNet()
test_pm(model)
