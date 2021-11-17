from data import Data_train
from data import Data_test
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from paddle.nn import Linear

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
Dataset_paddle_train = MyDataset(Data_train[:,:-10], Data_train[:,-10:])
Dataset_paddle_test = MyDataset(Data_test[:,:-10], Data_test[:,-10:])

train_loader = paddle.io.DataLoader(Dataset_paddle_train,
    batch_size=128,
    shuffle=True,
    use_buffer_reader=True,
    use_shared_memory=False,
    drop_last=True)

test_loader = paddle.io.DataLoader(Dataset_paddle_test,
    batch_size=256,
    shuffle=True,
    use_buffer_reader=True,
    use_shared_memory=False,
    drop_last=True)

# 分类网络
class Classification(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Classification, self).__init__()

        # 定义全连接层，输入维度是29，输出维度是10
        self.fc1 = Linear(in_features=29, out_features=22)
        self.Dr1 = paddle.nn.Dropout(p=0.1)
        self.fc2 = Linear(in_features=22, out_features=16)
        self.Dr2 = paddle.nn.Dropout(p=0.1)
        self.fc3 = Linear(in_features=16, out_features=10)

    # 网络的前向计算
    def forward(self, inputs):
        outputs1 = self.fc1(inputs)
        outputs1 = F.relu(outputs1)
        outputs1 = self.Dr1(outputs1)

        outputs2 = self.fc2(outputs1)
        outputs2 = F.sigmoid(outputs2)
        outputs2 = self.Dr2(outputs2)

        outputs3 = self.fc3(outputs2)
        outputs3 = F.sigmoid(outputs3)

        outputs_final = outputs3
        return outputs_final


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
            datas_train, labels_train = data_train_gather
            datas_train = paddle.to_tensor(datas_train, dtype='float32')
            labels_train = paddle.to_tensor(labels_train, dtype='float32')
            # 前向计算的过程
            predicts = model(datas_train)
            # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.binary_cross_entropy(predicts, labels_train)
            # 累计迭代次数和对应的loss
            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("train, epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss.numpy()))
                iters.append(iter)
                losses.append(loss.numpy())
                iter += 1

                test_losses = []

                for batch_kd, data_test_gather in enumerate(test_loader()):

                    datas_test, labels_test = data_test_gather
                    datas_test = paddle.to_tensor(datas_test, dtype='float32')
                    labels_test = paddle.to_tensor(labels_test, dtype='float32')

                    # 前向计算的过程
                    predicts_test = model(datas_test)
                    # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
                    test_loss = F.binary_cross_entropy(predicts_test, labels_test, reduction='mean')
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


#画出训练过程中Loss的变化曲线
plt.figure()
plt.title("train_val loss", fontsize=24)
plt.ylim((0, 1.0))
plt.xlabel("iter_train", fontsize=14)
plt.ylabel("loss_train", fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(iters_train, losses_train,color='red',label='train loss',linewidth=4)
plt.plot(iters_train, losses_test,color='blue',label='val loss',linewidth=4)
plt.grid()
plt.show()
