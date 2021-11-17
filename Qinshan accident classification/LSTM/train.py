#定义LSTM网络
import paddle
import paddle.nn.functional as F
from paddle.nn import Linear
from data import Data_train
from data import Data_test

# 建立LSTM时序数据格式
sequence_length = 19*1
delay = 1

# 定义数据读取类
class MyDataset(paddle.io.Dataset):
    def  __init__(self, feature_data, label_data):
        self.Feature_data = feature_data
        self.Label_data = label_data

    def __getitem__(self, idx):
        feature = self.Feature_data[idx]
        label = self.Label_data[idx]

        return feature, label

    def __len__(self):
        return len(self.Feature_data)
Dataset_paddle_train = MyDataset(Data_train[:,:,:-10], Data_train[:,:,-10:])
Dataset_paddle_test = MyDataset(Data_test[:,:,:-10], Data_test[:,:,-10:])

train_loader = paddle.io.DataLoader(Dataset_paddle_train,
    batch_size=128,
    shuffle=True,
    use_buffer_reader=True,
    use_shared_memory=False,
    drop_last=True)

test_loader = paddle.io.DataLoader(Dataset_paddle_test,
    batch_size=128,
    shuffle=True,
    use_buffer_reader=True,
    use_shared_memory=False,
    drop_last=True)


class  Classification(paddle.nn.Layer):
    def __init__(self):
        super(Classification,self).__init__()
        self.rnn = paddle.nn.LSTM(input_size=29,
                                  hidden_size=20,
                                  num_layers=1,
                                  time_major=False)
        # self.flatten = paddle.nn.Flatten()
        self.fc1 = Linear(20,14)
        self.fc2 = Linear(14,10)

    def forward(self,input):
        '''前向计算'''
        # print('input',input.shape)
        out, (h, c)=self.rnn(input)
        # 取LSTM输出序列的最后一个
        out = out[:,sequence_length+delay - 1,:]
        # out =self.flatten(out)
        out = F.relu(out)
        out=self.fc1(out)
        out = F.sigmoid(out)
        out=self.fc2(out)
        out = F.sigmoid(out)
        return out


def train(model):
    model.train()
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    opt = paddle.optimizer.Adam(learning_rate=0.0001, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),
                                parameters=model.parameters())
    EPOCH_NUM = 200
    iter = 0
    iters = []
    losses = []
    test_losses_mean = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data_train_gather in enumerate(train_loader()):
            datas_train, labels_train_un = data_train_gather
            labels_train = labels_train_un[:, sequence_length + delay - 1, :]
            datas_train = paddle.to_tensor(datas_train, dtype='float32')
            labels_train = paddle.to_tensor(labels_train, dtype='float32')
            # 前向计算的过程
            predicts = model(datas_train)
            # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.binary_cross_entropy(predicts, labels_train)
            # 累计迭代次数和对应的loss
            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 1200 == 0:
                print("train, epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss.numpy()))
                iters.append(iter)
                losses.append(loss.numpy())
                iter += 1

                test_losses = []

                for batch_kd, data_test_gather in enumerate(test_loader()):
                    datas_test, labels_test_un = data_test_gather
                    labels_test = labels_test_un[:, sequence_length + delay - 1, :]
                    datas_test = paddle.to_tensor(datas_test, dtype='float32')
                    labels_test = paddle.to_tensor(labels_test, dtype='float32')

                    # 前向计算的过程
                    predicts = model(datas_test)
                    # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
                    test_loss = F.binary_cross_entropy(predicts, labels_test, reduction='mean')
                    test_losses.append(test_loss)

                test_loss_mean = sum(test_losses) / len(test_losses)
                test_losses_mean.append(test_loss_mean.numpy())

            # 后向传播，更新参数的过程
            loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()
    # 保存模型参数
    paddle.save(opt.state_dict(), './Second classification/Classification.pdopt')
    paddle.save(model.state_dict(), './Second classification/Classification.pdparams')
    return iters, losses, test_losses_mean


# 创建模型
model = Classification()
# 启动训练过程
iters_train, losses_train, losses_test = train(model)

import matplotlib.pyplot as plt
#画出训练过程中Loss的变化曲线
plt.figure()
plt.title("train_val loss", fontsize=24)
plt.ylim((0, 1.0))
plt.xlabel("iter_train", fontsize=14)
plt.ylabel("loss_train", fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(iters_train, losses_train,color='red',label='train loss',linewidth=4)
plt.plot(iters_train, losses_test,color='black',label='val loss',linewidth=4)
plt.grid()
plt.show()