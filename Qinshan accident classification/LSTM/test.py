import paddle
import paddle.nn.functional as F
from paddle.nn import Linear
import numpy as np
from data import Data_test

# 建立LSTM时序数据格式
sequence_length = 19*1
delay = 1

class  Classification(paddle.nn.Layer):
    '''
    LSTM网络
    '''
    def __init__(self):
        super(Classification,self).__init__()
        self.rnn = paddle.nn.LSTM(input_size=29, hidden_size=20, num_layers=1, time_major=False)
        # self.flatten = paddle.nn.Flatten()
        self.fc1 = Linear(20,14)
        self.fc2 = Linear(14,10)

    def forward(self,input):        # forward 定义执行实际运行时网络的执行逻辑
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

# 创建模型
model = Classification()
# 模型效果展示
params_dict = paddle.load('./Second classification/Classification.pdparams')
opt_dict = paddle.load('./Second classification/Classification.pdopt')
# 加载参数到模型
model.set_state_dict(params_dict)
opt = paddle.optimizer.Adam(learning_rate=0.0001, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5), parameters=model.parameters())
opt.set_state_dict(opt_dict)

results_predict_data = paddle.to_tensor(Data_test[:,:,:-10], dtype = 'float32')
results_predict = model(results_predict_data)
results_predict_np = np.array(results_predict)
results_predict_Max_index = np.argmax(results_predict_np,axis=1)

results_label_data = Data_test[:,sequence_length+delay - 1,-10:]
results_label_Max_index = np.argmax(results_label_data,axis=1)
k = 0
for i in range(len(results_predict_Max_index)):
    if results_predict_Max_index[i] == results_label_Max_index[i]:
        k += 1
accuracy_rate = k / len(results_predict_Max_index)
print('模型测试的预测准确率为 {}'.format(accuracy_rate))

