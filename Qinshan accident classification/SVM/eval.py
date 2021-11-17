from data import x_data
from data import y_data

import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sklearn.model_selection as model_selection
from sklearn.metrics import classification_report
file = r'D:\Pycharm 2020\Projects\PaddlePaddle 2.1\QinShan\SVM\model save\SVM.joblib_031954'

X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.80, test_size=0.20, random_state=101)

# 读取模型
svm_model = joblib.load(file)

rbf_pred = svm_model.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.8f" % (rbf_accuracy))
print('F1 (RBF Kernel): ', "%.8f" % (rbf_f1))
print(classification_report(y_test, rbf_pred))
