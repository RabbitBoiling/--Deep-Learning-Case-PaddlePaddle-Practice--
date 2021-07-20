import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(selected_data):
    # 为缺失Age记灵境和彷设置为平均僮
    age_mean_value = selected_data['Age'].mean()
    selected_data.loc[:,'Age'] = selected_data.loc[:,'Age'].fillna(age_mean_value)
    # 为缺失Fare记灵境究危
    fare_mean_value = selected_data['Fare'].mean()
    selected_data.loc[:,'Fare'] = selected_data.loc[:,'Fare'].fillna(fare_mean_value)
    # 为缺失Embarked记灵境芫僮
    selected_data.loc[:,'Embarked'] = selected_data.loc[:,'Embarked'].fillna('S')

    # sex字符串转痪为数字编码
    selected_data.loc[:,'Sex'] = selected_data.loc[:,'Sex'].map({'female': 0, 'male': 1}).astype(int)
    # 港口embarked团字母表示转唤为数字编码
    selected_data.loc[:,'Embarked'] = selected_data.loc[:,'Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    # drop不改变原有的df中的数据，而是返回另一个DataFrame来存放删除后的数据,axis = 1 表示删除列
    selected_df_data = selected_data.drop(['Name'], axis=1)

    # 转挽为ndarray数绢
    ndarray_data = selected_df_data.values
    ndarray_data = ndarray_data.astype(np.float32) # 必须由float64转化为float32
    # 特彶詹标淮化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_ndarray_data = minmax_scale.fit_transform(ndarray_data)

    return norm_ndarray_data

# 得到处理后的训练集
# 读取数据文件，结果为DataFrame格式
data_file_path = "./train.xlsx"
df_data = pd.read_excel(data_file_path)
# 筛选提取需要的待彶字段，去葆ticket, cabin等
selected_cols = ['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
selected_df_data = df_data[selected_cols]
# shuffle, 打刮数据顺序，通过Pandas的拙样函数sample实现frac为百分比
shuffled_df_data = selected_df_data.sample(frac=1)
train_data = prepare_data(selected_data = shuffled_df_data)

# 得到处理后的测试集
data_file_path = "./train.xlsx"
df_data = pd.read_excel(data_file_path)
selected_cols = ['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
selected_df_data = df_data[selected_cols]
shuffled_df_data = selected_df_data.sample(frac=1)
test_data = prepare_data(selected_data = shuffled_df_data)

# 加载数据
training_data, test_data = train_data, test_data

class Regressor(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()

        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc1 = Linear(in_features=7, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=32)
        self.fc3 = Linear(in_features=32, out_features=1)

    # 网络的前向计算
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

# 声明定义好的线性回归模型
model = Regressor()
# 定义训练过程
def train(model):
    paddle.set_device('gpu:0')
    print('start training .......')
    # 开启模型训练模式
    model.train()
    # 定义优化算法，使用随机梯度下降SGD
    # 学习率设置为0.01
    opt = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())

    EPOCH_NUM = 300  # 设置外层循环次数
    BATCH_SIZE = 32  # 设置batch大小
    epoch = 0
    epochs = []
    train_losses = []
    train_losses_set = []
    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 将训练数据进行拆分，每个batch包含10条数据
        mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            x = mini_batch[:, 1:]  # 后7叨悬特征列，二维数组
            y = mini_batch[:, 0]  # 第0列是标签列，一维数组
            y = y.reshape(len(y),1) #最后一个批次可能不是BATCH_SIZE大小
            # 将numpy数据转为飞桨动态图tensor形式
            Survival_features = paddle.to_tensor(x)
            Survival_probability = paddle.to_tensor(y)
            # 前向计算
            predicts = model(Survival_features)

            # 计算损失，采用二元交叉熵损失
            loss = F.binary_cross_entropy(predicts, label=Survival_probability)
            avg_loss = paddle.mean(loss)
            train_losses.append(avg_loss.numpy())
            if iter_id % 20 == 0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

        # 求平均精度
        train_losses_mean = np.array(train_losses).mean()
        train_losses_set.append(train_losses_mean)
        epoch = epoch + 1
        epochs.append(epoch)

    # 保存模型参数，文件名为LR_model.pdparams
    paddle.save(model.state_dict(), 'LR_model.pdparams')
    print("模型保存成功，模型参数保存在LR_model.pdparams中")

    return epochs, train_losses_set

# 定义评估过程
def evaluation(model):
    paddle.set_device('gpu:0')
    print('start evaluation .......')
    # 参数为保存模型参数的文件地址
    model_dict = paddle.load('LR_model.pdparams')
    model.load_dict(model_dict)
    model.eval()

    BATCH_SIZE = 32  # 设置batch大小
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(test_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [test_data[k:k + BATCH_SIZE] for k in range(0, len(test_data), BATCH_SIZE)]
    # 定义内层循环
    acc_set = []
    avg_loss_set = []
    for iter_id, mini_batch in enumerate(mini_batches):
        x = mini_batch[:, 1:]  # 后7叨悬特征列，二维数组
        y = mini_batch[:, 0]  # 第0列是标签列，一维数组
        y = y.reshape(len(y),1) #最后一个批次可能不是BATCH_SIZE大小

        y = y.astype(np.int64)
        # 将numpy数据转为飞桨动态图tensor形式
        Survival_features = paddle.to_tensor(x)
        Survival_probability_int64 = paddle.to_tensor(y)
        # 计算预测和精度
        prediction = model(Survival_features)
        acc = paddle.metric.accuracy(prediction, Survival_probability_int64)
        # 计算损失，采用二元交叉熵损失
        y = y.astype(np.float32)
        Survival_probability_float32 = paddle.to_tensor(y)
        loss = F.binary_cross_entropy(prediction, label=Survival_probability_float32)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    # 求平均精度
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('模型在测试集的评估结果为：loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))


model = Regressor()
# 模型训练
Epochs, Train_loss = train(model)
# 模型评估
evaluation(model)

# 画出训练过程中Loss的变化曲线
plt.figure()
plt.title("Train loss", fontsize=24)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.plot(Epochs, Train_loss, color='red', label='train loss')
plt.grid()
plt.show()

Jack_Rose = np.array(([0, 3, 1, 23, 1, 0, 5.0000, 2],[1, 1, 0, 20, 1, 0, 100.0000, 2]),dtype=np.float32)
verify_data = np.append(training_data,Jack_Rose,axis=0)
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
verify_data = minmax_scale.fit_transform(verify_data)
verify_data = verify_data[:, 1:]
verify_data = paddle.to_tensor(verify_data)
model = Regressor()
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()
Survival_prediction = model(verify_data)
Survival_prediction = np.array(Survival_prediction) # 将tensor格式转化为numpy格式
print('Jack与Rose的生存概率分别为{},{}'.format(Survival_prediction[891,0],Survival_prediction[892,0]))
