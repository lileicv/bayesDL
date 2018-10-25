import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(xtr, ytr, xte, yte, imname):
    print(xtr.shape, ytr.shape, xte.shape, yte.shape)
    xtr = np.squeeze(xtr)
    ytr = np.squeeze(ytr)
    xte = np.squeeze(xte)
    yte = np.squeeze(yte)
    plt.figure(0)
    plt.plot(xtr, ytr, 'r.')
    plt.plot(xte, yte, 'b-')
    plt.savefig(imname)
    plt.close()

# Create toy dataset
n_sample = 100
X = np.random.normal(size=(n_sample,1))
Y = np.random.normal(np.cos(5.*X)/(np.abs(X)+1.),0.1)
X_pred = np.atleast_2d(np.linspace(-3,3,num=100)).T

# Create Model
X_ = tf.placeholder(tf.float32, shape=[None, 1])
Y_ = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([1, 100],stddev=0.35), name='W1')
b1 = tf.Variable(tf.zeros([100]), name='b1')
fc1 = tf.nn.relu(tf.matmul(X_,W1)+b1)
W2 = tf.Variable(tf.random_normal([100,100],stddev=0.35), name='W2')
b2 = tf.Variable(tf.zeros([100]), name='b2')
fc2 = tf.nn.relu(tf.matmul(fc1,W2)+b2)
W3 = tf.Variable(tf.random_normal([100,1],stddev=0.35), name='W3')
b3 = tf.Variable(tf.zeros([1]), name='b3')
fc3 = tf.matmul(fc2,W3)+b3
'''
fc1 = tf.layers.dense(X_, 100, activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2, 1)
'''
loss = tf.reduce_sum(tf.abs(Y_-fc3))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
mse = tf.reduce_sum(tf.square(Y_-fc3))/n_sample

# Train the model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(30000):
    [tr_mse, _] = sess.run([mse, train_step], feed_dict={X_:X, Y_:Y})
    if i%100 == 0:
        print('train step:{}, train_loss:{}'.format(i, tr_mse))
    if i%1000 == 0:
        Y_pred = sess.run(fc3, feed_dict={X_:X_pred})
        draw(X,Y,X_pred,Y_pred, './rstmlp/{}.jpg'.format(i))
