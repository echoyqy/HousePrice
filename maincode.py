import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.utils import shuffle

from sklearn.preprocessing import normalize
# sklearn 随机打乱工具

df = pd.read_csv('/Volumes/yqy/python/HousePrice/data/boston.csv')
# print(df.describe())
# print(type(df))
# 当前数据为class类型，要处理的话得运行为数组形式
df = df.values
df = np.array(df)
x_data = df[:,:12]
x_data = x_data / x_data.max(axis=0)
y_data = df[:,12]
y_data = y_data / y_data.max(axis=0)
# 定义占位
x = tf.placeholder(tf.float32,[None,12],name='x')
y = tf.placeholder(tf.float32,[None,1],name='y')
# 定义模型结构

with tf.name_scope('Model'):
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name='W')
    b = tf.Variable(1.0, name='b')
    def model(w,x,b):
        return tf.multiply(x,w)+b
    pred = model(w,x,b)
    train_epoches = 100
    learning_rate = 0.0001
    with tf.name_scope('Lossfunction'):
        Lossfunction = tf.reduce_mean(tf.pow(y-pred,2))
# 计算均方误差
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(Lossfunction)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for epoch in range(train_epoches):
    loss_sum=0.0
    for xs,ys in zip(x_data, y_data):
        xs = xs.reshape(1, 12)
        ys = ys.reshape(1,1)
        _,loss=sess.run([optimizer, Lossfunction], feed_dict={x: xs, y: ys} )
        loss_sum = loss + loss_sum
    x_data, y_data = shuffle(x_data, y_data)
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum/len(y_data)
    print('epoch=', epoch+1,'loss=', loss_average, 'b=', b0temp, 'w=', w0temp )



