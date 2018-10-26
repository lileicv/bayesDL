''' Build Bayesian Neural Network with Tensorflow Probability
    Author: Lilei
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

def draw(xtr, ytr, xte, yte, imname):
    xtr = np.squeeze(xtr)
    ytr = np.squeeze(ytr)
    xte = np.squeeze(xte)
    yte = np.squeeze(yte)
    print(xtr.shape, ytr.shape, xte.shape, yte.shape)
    plt.figure(0)
    plt.plot(xtr, ytr, 'r.')
    for j in range(yte.shape[0]):
        plt.plot(xte, yte[j], 'b-', alpha=0.1)
    plt.savefig(imname)
    plt.close()

# Make toy dataset
n_sample = 100
X = np.random.normal(size=(n_sample,1))
Y = np.random.normal(np.cos(5.*X)/(np.abs(X)+1.), 0.1)
X_pred = np.atleast_2d(np.linspace(-3,3,num=100)).T

# Build the model
X_ = tf.placeholder(tf.float32, shape=[None, 1])
Y_ = tf.placeholder(tf.float32, shape=[None, 1])
x_in = tf.keras.Input(tensor=X_)
x = tfp.layers.DenseFlipout(128,activation='relu')(x_in)
x = tfp.layers.DenseFlipout(128,activation='relu')(x)
x_out = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=x_in,outputs=x_out)

norm = tfp.distributions.Normal(Y_, 0.1)
log_likelihood = tf.reduce_sum(norm.log_prob(x_out))
kl = sum(model.losses)
elbo_loss = -log_likelihood + kl

train_op = tf.train.AdamOptimizer(0.001).minimize(elbo_loss)
mse = tf.reduce_sum(tf.square(Y_-x_out))/n_sample

# Train the model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(50000):
    [tr_mse, _] = sess.run([mse,train_op], feed_dict={X_:X, Y_: Y})
    if i % 100 == 0:
        print('train step:{}, train loss:{}'.format(i, tr_mse))
    if i% 1000 == 0:
        Y_pred = np.zeros([20, X_pred.shape[0]])
        for j in range(20):
            Y_pred[j,:] = np.squeeze(sess.run(x_out, feed_dict={X_:X_pred}))
        draw(X, Y, X_pred, Y_pred, './rst_tfp/{}.jpg'.format(i))
    
