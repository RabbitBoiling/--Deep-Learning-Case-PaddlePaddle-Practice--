from data import x_data
from data import y_data

import warnings
warnings.filterwarnings("ignore") # 代码运行警告过滤

import time
import joblib
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.80, test_size=0.20, random_state=101)
rbf = svm.SVC(C=0.1, kernel='rbf', gamma=0.1, tol=0.00001).fit(X_train, y_train) #高斯核
rbf_pred = rbf.predict(X_test)

# 给保存的模型的名字加上时间标签，以区分训练过程中产生的不同的模型
mdhms = time.strftime('%d%H%M', time.localtime(time.time()))
# 保存的模型的文件名
file = r'D:\Pycharm 2020\Projects\PaddlePaddle 2.1\QinShan\SVM\model save\SVM.joblib' + '_' + mdhms
# 保存模型
joblib.dump(rbf,file)

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.8f" % (rbf_accuracy))
print('F1 (RBF Kernel): ', "%.8f" % (rbf_f1))
print(classification_report(y_test, rbf_pred))

