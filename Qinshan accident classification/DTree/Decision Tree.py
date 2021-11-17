from data import x_data
from data import y_data

import warnings
warnings.filterwarnings("ignore") # 代码运行警告过滤

from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.80, test_size=0.20, random_state=101)
tree = DecisionTreeClassifier(criterion='gini', splitter='random').fit(X_train, y_train) # 决策树
tree_pred = tree.predict(X_test)

rbf_accuracy = accuracy_score(y_test, tree_pred)
rbf_f1 = f1_score(y_test, tree_pred, average='weighted')
print('Accuracy (Decision Tree): ', "%.8f" % (rbf_accuracy))
print('F1 (Decision Tree): ', "%.8f" % (rbf_f1))
print(classification_report(y_test, tree_pred))

