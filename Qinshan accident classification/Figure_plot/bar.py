import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

X = ['KNN算法', '决策树算法', 'SVM算法', '深度前馈神经网络', 'LSTM神经网络']
Y = [0.99892428, 0.99904381, 0.95673221, 0.90838, 0.99952]
fig = plt.figure()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(X, Y, 0.4, color="blue")
plt.xlabel("预测分类算法", fontsize=20)
plt.ylabel("测试集的准确率", fontsize=20)
plt.title("不同的算法在测试集上的分类准确性对比", fontsize=20)
plt.rcParams.update({'font.size': 25})


plt.show()