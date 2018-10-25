import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(xtr, ytr, xte, yte, imname):
    xtr = np.squeeze(xtr)
    ytr = np.squeeze(ytr)
    xte = np.squeeze(xte)
    yte = np.squeeze(yte)
    print(xtr.shape, ytr.shape, xte.shape, yte.shape)
    plt.figure()
    plt.plot(xtr, ytr, 'r.')
    for j in range(yte.shape[0]):
        plt.plot(xte, yte[j], 'b-', alpha=0.1)
    plt.savefig(imname)

def log_gaussian(x, mu, sigma):
    return -0.5*tf.log(2.*np.pi)-tf.log(sigma)-(x-mu)**2/(2*sigma**2)

class VariationalDense:
    def __init__(self, n_in, n_out):
        self.W_mu = tf.Variable(tf.random_normal([n_in, n_out], stddev=0.35))
        self.W_logsigma = tf.Variable(tf.truncated_normal([n_in, n_out], mean=0, stddev=0.35))
        self.b_mu = tf.Variable(tf.zeros([n_out]))
        self.b_logsigma = tf.Variable(tf.zeros([n_out]))

        epsilon_w = tf.random_normal([n_in, n_out], mean=0., stddev=0.1)
        epsilon_b = tf.random_normal([n_out], mean=0., stddev=0.1)
        #self.W = self.W_mu + tf.multiply(tf.log(1.+tf.exp(self.W_logsigma)), epsilon_w)
        #self.b = self.b_mu + tf.multiply(tf.log(1.+tf.exp(self.b_logsigma)), epsilon_b)
        self.W = self.W_mu + tf.multiply(tf.exp(self.W_logsigma), epsilon_w)
        self.b = self.b_mu + tf.multiply(tf.exp(self.b_logsigma), epsilon_b)

    def __call__(self, x, activation=tf.identity):
        output = activation(tf.matmul(x, self.W)+self.b)
        #output = tf.squeeze(output)
        return output

    def regularization(self):
        log_pw, log_qw = 0., 0.
        log_pw += tf.reduce_sum(log_gaussian(self.W, 0., 1.))
        log_pw += tf.reduce_sum(log_gaussian(self.b, 0., 1.))
        #log_qw += tf.reduce_sum(log_gaussian(self.W, self.W_mu, tf.log(1+tf.exp(self.W_logsigma))))
        #log_qw += tf.reduce_sum(log_gaussian(self.b, self.b_mu, tf.log(1+tf.exp(self.b_logsigma))))
        log_qw += tf.reduce_sum(log_gaussian(self.W, self.W_mu, tf.exp(self.W_logsigma)))
        log_qw += tf.reduce_sum(log_gaussian(self.b, self.b_mu, tf.exp(self.b_logsigma)))
        return tf.reduce_sum(log_qw-log_pw)

# Build toy dataset
n_sample = 100
X = np.random.normal(size=(n_sample,1))
Y = np.random.normal(np.cos(5.*X)/(np.abs(X)+1.), 0.1)
X_pred = np.atleast_2d(np.linspace(-3,3,num=100)).T

# Create the Model
X_ = tf.placeholder(tf.float32, shape=[None, 1])
Y_ = tf.placeholder(tf.float32, shape=[None, 1])
lr_ = tf.placeholder(tf.float32)
layer_1 = VariationalDense(1, 100)
layer_2 = VariationalDense(100, 100)
layer_3 = VariationalDense(100, 1)

f1 = layer_1(X_, tf.nn.relu)
f2 = layer_2(f1, tf.nn.relu)
f3 = layer_3(f2)

log_likelihood = tf.reduce_sum(log_gaussian(Y_, f3, 0.1))
regularization = layer_1.regularization() + layer_2.regularization() + layer_3.regularization()
KL = -log_likelihood + regularization/100
train_step = tf.train.AdamOptimizer(lr_).minimize(KL)
mse = tf.reduce_sum(tf.square(Y_-f3))/n_sample

# Train the model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(30000):
    lr = 0.1**(2+i//10000)
    [tr_mse, _] = sess.run([mse,train_step], feed_dict={X_:X, Y_: Y, lr_:lr})
    if i % 100 == 0:
        print('train step:{}, train loss:{}, lr:{}'.format(i, tr_mse, lr))
    if i% 1000 == 0:
        Y_pred = np.zeros([20, X_pred.shape[0]])
        for j in range(20):
            Y_pred[j,:] = np.squeeze(sess.run(f3, feed_dict={X_:X_pred}))
        draw(X, Y, X_pred, Y_pred, './rstvi/{}.jpg'.format(i))
