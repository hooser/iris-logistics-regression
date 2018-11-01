import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test =train_test_split(X,y,train_size=0.3,random_state=1)
#调节参数
# c=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
solver=['lbfgs','newton-cg','liblinear','sag','saga']
loss = []
for j in solver:

    # for i in c:
    clf = LogisticRegression(penalty='l2', solver=j)
    clf.fit(x_train,y_train)
    pre=clf.predict_proba(x_test)#不能用predict  因为proba概率形式
    loss.append(log_loss(y_test,pre))
plt.plot(solver, loss)
    # plt.title("solver="+j)
plt.xlabel("solver")  # X轴标签
plt.ylabel("loss")  # Y轴标签
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#多分类
for i in solver:
    clf = LogisticRegression(solver='lbfgs',multi_class='multinomial')
    clf.fit(x_train,y_train)
    pre=clf.predict_proba(x_test)#不能用predict  因为proba概率形式
    loss.append(log_loss(y_test, pre))
plt.plot(solver, loss)
plt.xlabel("solver")  # X轴标签
plt.ylabel("loss")  # Y轴标签
plt.show()
# print(loss)