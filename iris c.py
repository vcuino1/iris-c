from sklearn import svm

import numpy as np

from sklearn import model_selection

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import colors

def iris_type(s):

    class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}

    return class_label[s]

filepath='C:/Users/崔亚祺/Downloads/iris.data'  # 数据文件路径

data=np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})

X ,y=np.split(data,(4,),axis=1) 

x=X[:,0:2]

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)

classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)

classifier.fit(x_train,y_train.ravel())

def show_accuracy(y_hat,y_train,str):

    pass

print("SVM-输出训练集的准确率为：",classifier.score(x_train,y_train))

y_hat=classifier.predict(x_train)

show_accuracy(y_hat,y_train,'训练集')

print("SVM-输出测试集的准确率为：",classifier.score(x_test,y_test))

y_hat=classifier.predict(x_test)

show_accuracy(y_hat,y_test,'测试集')

#print('\npredict:\n', classifier.predict(x_train))

x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  

x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  

x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j] 

grid_test = np.stack((x1.flat, x2.flat), axis=1)  

#print("grid_test = \n", grid_test)

grid_hat = classifier.predict(grid_test)   

#print("grid_hat = \n", grid_hat)

grid_hat = grid_hat.reshape(x1.shape)  

mpl.rcParams['font.sans-serif'] = [u'SimHei']

mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])

cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

alpha=0.5

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light) 

plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')

plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10) 

plt.xlabel(u'花萼长度', fontsize=13)

plt.ylabel(u'花萼宽度', fontsize=13)

plt.xlim(x1_min, x1_max)

plt.ylim(x2_min, x2_max)

plt.title(u'鸢尾花SVM二特征分类', fontsize=15)

plt.show()