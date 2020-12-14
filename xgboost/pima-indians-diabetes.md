[How to Develop Your First XGBoost Model in Python with scikit-learn](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/)

[xgboost python-api](http://xgboost.readthedocs.io/en/latest/python/python_api.html)

二分类问题，判断病人是否会在5年内患糖尿病，前8列是变量，最后一列是预测值为0或1。
# 1 基础应用
## 导入相关包
```
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
## 加载数据
[下载测试数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)
```
dataset = loadtxt('pima-indians-diabetes.data', delimiter=",")
```
## 分出变量和标签
```
X = dataset[:, 0:8]
Y = dataset[:, 8]
```
## 将数据分为训练集和测试集
训练集用来学习模型，测试集用来预测
```
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=test_size, random_state=seed)
```
## 用XGBClassifier建立模型
[XGBClassifier](http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
```
model = XGBClassifier()
model.fit(X_train, y_train)
```
## 结果
xgboost 的结果是每个样本属于第一类的概率，需要用 round 将其转换为 0 1 值, Accuracy: 77.95%。
```
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```
# 2 监控模型表现
xgboost 可以在模型训练时，评价模型在测试集上的表现，也可以输出每一步的分数。
```
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train,
          y_train,
          early_stopping_rounds=10,
          eval_metric="logloss",
          eval_set=eval_set,
          verbose=True)
```
那么它会在每加入一颗树后打印出 logloss：
```
[1]	validation_0-logloss:0.634854
......
[38]	validation_0-logloss:0.489334
[39]	validation_0-logloss:0.490969
[40]	validation_0-logloss:0.48978
[41]	validation_0-logloss:0.490704
[42]	validation_0-logloss:0.492369
```
并打印出 Early Stopping 的点：
```
Stopping. Best iteration:
[32]	validation_0-logloss:0.487297
```
# 3 输出特征重要度
gradient boosting 还有一个优点是可以给出训练好的模型的特征重要性，这样就可以知道哪些变量需要被保留，哪些可以舍弃。
```
from xgboost import plot_importance
from matplotlib import pyplot
......
plot_importance(model)
pyplot.show()
......
```
![pima-indians-diabetes特征重要度](http://note.youdao.com/yws/api/personal/file/8457ACFCD22F4AA391FEAD4D3B48AF0C?method=download&shareKey=69f3cb2b0759bed0c5d6c1958e797698)
# 4 调参
如何调参呢，下面是三个超参数的一般实践最佳值，可以先将它们设定为这个范围，然后画出 learning curves，再调解参数找到最佳模型：
- learning_rate ＝ 0.1 或更小，越小就需要多加入弱学习器；
- tree_depth ＝ 2～8；
- subsample ＝ 训练集的 30%～80%；

接下来我们用 GridSearchCV 来进行调参会更方便一些。

可以调的超参数组合有：
- 树的个数和大小 (n_estimators and max_depth).
- 学习率和树的个数 (learning_rate and n_estimators).
- 行列的 subsampling rates (subsample, colsample_bytree and colsample_bylevel).
# 下面以学习率为例：
先引入这两个类：
```
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
```
设定要调节的 learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]：
```
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(
    model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```
最后会给出最佳的学习率为 0.1：
```
Best: -0.483013 using {'learning_rate': 0.1}
```
我们还可以用下面的代码打印出每一个学习率对应的分数：
```
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```
```
-0.689650 (0.000242) with: {'learning_rate': 0.0001}
-0.661274 (0.001954) with: {'learning_rate': 0.001}
-0.530747 (0.022961) with: {'learning_rate': 0.01}
-0.483013 (0.060755) with: {'learning_rate': 0.1}
-0.515440 (0.068974) with: {'learning_rate': 0.2}
-0.557315 (0.081738) with: {'learning_rate': 0.3}
```
# 完整测试程序
```
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def main():
    dataset = loadtxt('pima-indians-diabetes.data', delimiter=",")

    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed)

    model = XGBClassifier()
    eval_set = [(X_test, y_test)]
    model.fit(X_train,
              y_train,
              early_stopping_rounds=10,
              eval_metric="logloss",
              eval_set=eval_set,
              verbose=True)

    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(
        model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, Y)

    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    plot_importance(model)
    pyplot.show()


if __name__ == "__main__":
    main()
```
# 参考
