import paddle
import paddle.nn.functional as F
from paddle.nn import Linear
import numpy as np
from data import Data_test

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

# 创建模型
model = Classification()
# 模型效果展示
params_dict = paddle.load('./Second classification/Classification.pdparams')
opt_dict = paddle.load('./Second classification/Classification.pdopt')
# 加载参数到模型
model.set_state_dict(params_dict)
opt = paddle.optimizer.Adam(learning_rate=0.0001, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5), parameters=model.parameters())
opt.set_state_dict(opt_dict)

results_predict_data = paddle.to_tensor(Data_test[:,:-10], dtype = 'float32')
results_predict = model(results_predict_data)
results_predict_np = np.array(results_predict)
results_predict_Max_index = np.argmax(results_predict_np,axis=1)

results_label_data = Data_test[:,-10:]
results_label_Max_index = np.argmax(results_label_data,axis=1)
k = 0
for i in range(len(results_predict_Max_index)):
    if results_predict_Max_index[i] == results_label_Max_index[i]:
        k += 1
accuracy_rate = k / len(results_predict_Max_index)
print('模型测试的预测准确率为 {}'.format(accuracy_rate))

