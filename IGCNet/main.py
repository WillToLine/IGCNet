# %%
import scipy.io as sio  # import scipy.io for .mat file I/O
import numpy as np  # import numpy
import matplotlib.pyplot as plt  # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol as wf
import tensorflow as tf

# %%
K = 20  # number of users
num_H = 2000  # number of training samples
num_test = 500  # number of testing  samples
training_epochs = 50  # number of training epochs
trainseed = 0  # set random seed for training set
testseed = 7  # set random seed for test set
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n' % (K, num_H, training_epochs))
var_db = 0
var = 1 / 10 ** (var_db / 10)
# %%
import time


def generate_wGaussian(K, num_H, var_noise=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pmax = 1
    Pini = Pmax * np.ones(K)
    # alpha = np.random.rand(num_H,K)
    alpha = np.ones((num_H, K))
    # var_noise = 1
    X = np.zeros((K ** 2, num_H))
    Y = np.zeros((K, num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1 / np.sqrt(2) * (np.random.randn(K, K) + 1j * np.random.randn(K, K))
        H = abs(CH)
        X[:, loop] = np.reshape(H, (K ** 2,), order="F")
        H = np.reshape(X[:, loop], (K, K), order="F")
        mid_time = time.time()
        Y[:, loop] = wf.WMMSE_sum_rate2(Pini, alpha[loop, :], H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time

    # print("wmmse time: %0.2f s" % total_time)
    return X, Y, alpha, total_time


# %%
Xtrain, Ytrain, Atrain, wtime = generate_wGaussian(K, num_H, seed=trainseed, var_noise=var)
X, Y, A, wmmsetime = generate_wGaussian(K, num_test, seed=testseed, var_noise=var)
Xtrain = Xtrain.transpose()
X = X.transpose()
Ytrain = Ytrain.transpose()
Y = Y.transpose()
print(Xtrain.shape, Ytrain.shape)
# %%
Xtrain = Xtrain.reshape((-1, K, K))
X = X.reshape((-1, K, K))


# %%
def extract_features(H, n, L, alpha):
    direct_H = np.zeros((n, L))
    inter_to = np.zeros((n, L, L))
    inter_from = np.zeros((n, L, L))
    other_H = np.zeros((n, L, L))
    for ii in range(n):
        diag_H = np.diag(H[ii, :, :])
        for jj in range(L):
            direct_H[ii, jj] = H[ii, jj, jj]
            inter_to[ii, jj, :] = H[ii, :, jj].T
            inter_to[ii, jj, jj] = 0
            inter_from[ii, jj, :] = H[ii, jj, :]
            inter_from[ii, jj, jj] = 0
            other_H[ii, jj, :] = diag_H
            other_H[ii, jj, jj] = 0
    return direct_H, inter_to, inter_from, other_H, alpha


features = extract_features(Xtrain, num_H, K, Atrain)


# %%
def extract_labels(y, n, L):
    labels = np.zeros((n, L, 2))
    for ii in range(n):
        for jj in range(L):
            if (abs(y[ii][jj]) < 1e-4):
                labels[ii, jj, :] = [1, 0]
            else:
                labels[ii, jj, :] = [0, 1]
    return labels


labels = extract_labels(Ytrain, num_H, K)
labels_t = extract_labels(Y, num_test, K)
# %%
weights = {
    'wif1': tf.Variable(tf.random_normal([1, 1, 5, 16], stddev=0.1)),
    'wif2': tf.Variable(tf.random_normal([1, 1, 16, 16], stddev=0.1)),
    'wif3': tf.Variable(tf.random_normal([1, 1, 16, 6], stddev=0.1)),

    'wfc1': tf.Variable(tf.random_normal([15, 32], stddev=0.1)),
    'wfc2': tf.Variable(tf.random_normal([32, 16], stddev=0.1)),
    'wfc3': tf.Variable(tf.random_normal([16, 1])),
}

biases = {
    'bif1': tf.Variable(tf.random_normal([16], stddev=0.1)),
    'bif2': tf.Variable(tf.random_normal([16], stddev=0.1)),
    'bif3': tf.Variable(tf.random_normal([6], stddev=0.1)),

    'bfc1': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'bfc2': tf.Variable(tf.random_normal([16], stddev=0.1)),
    'bfc3': tf.Variable(tf.random_normal([1])),

}


# %%
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def inter_conv_net(Ws, bs, inter):
    h1 = conv2d(inter, Ws[0], bs[0])
    h2 = conv2d(h1, Ws[1], bs[1])
    h3 = conv2d(h2, Ws[2], bs[2])
    fea_mean = tf.reduce_sum(h3, axis=2)
    fea_max = tf.reduce_max(h3, axis=2)
    fea = tf.concat([fea_mean, fea_max], axis=2)
    return fea


def fully_connected_net(Ws, bs, feat):
    hidden1 = tf.nn.relu(tf.tensordot(feat, Ws[0], [[2], [0]]) + bs[0])
    hidden2 = tf.nn.relu(tf.tensordot(hidden1, Ws[1], [[2], [0]]) + bs[1])
    out = tf.tensordot(hidden2, Ws[2], [[2], [0]]) + bs[2]
    return out


# %%
L = K
Xinterf = tf.placeholder(tf.float32, [None, L, L, 1])
Htrain = tf.placeholder(tf.float32, [None, L, L])
Xintert = tf.placeholder(tf.float32, [None, L, L, 1])
Xdiag = tf.placeholder(tf.float32, [None, L, 1])
Xdiag_o = tf.placeholder(tf.float32, [None, L, L, 1])
intensity = tf.placeholder(tf.float32, [None, L, 1])
w_alpha = tf.placeholder(tf.float32, [None, L, 1])
y_ = tf.placeholder(tf.float32, [None, L, 1])

Winterf = [weights['wif1'], weights['wif2'], weights['wif3']]
binterf = [biases['bif1'], biases['bif2'], biases['bif3']]

Wfc = [weights['wfc1'], weights['wfc2'], weights['wfc3']]
bfc = [biases['bfc1'], biases['bfc2'], biases['bfc3']]
# %%
intens = intensity

all_one = tf.ones((1, L), dtype=tf.float32)
intens2 = tf.tensordot(intens, all_one, [[2], [0]])  # tf.matmul(intens,all_one)
w2 = tf.tensordot(w_alpha, all_one, [[2], [0]])

intens2 = tf.transpose(intens2, perm=[0, 2, 1])
intens2 = tf.reshape(intens2, (-1, L, L, 1))

w2 = tf.transpose(w2, perm=[0, 2, 1])
w2 = tf.reshape(w2, (-1, L, L, 1))

w_alpha2 = tf.reshape(w_alpha, (-1, L))

xf = tf.concat((Xinterf, Xintert, Xdiag_o, intens2, w2), axis=3)

for ii in range(5):
    fea1 = inter_conv_net(Winterf, binterf, xf)
    fea = tf.concat([Xdiag, intens, fea1, w_alpha], axis=2)  #
    out = fully_connected_net(Wfc, bfc, fea)
    pred = tf.nn.sigmoid(out)
    intens = pred
    intens2 = tf.tensordot(intens, all_one, [[2], [0]])
    intens2 = tf.transpose(intens2, perm=[0, 2, 1])
    intens2 = tf.reshape(intens2, (-1, L, L, 1))
    xf = tf.concat((Xinterf, intens2, Xintert, Xdiag_o, w2), axis=3)

pred = tf.reshape(pred, (-1, K))
H = tf.math.square(Htrain)
H_diag = tf.matrix_diag_part(H)
fr = H_diag * pred
pred2 = tf.reshape(pred, [-1, K, 1])
ag = tf.reshape(tf.matmul(H, pred2), [-1, K]) + var - fr
obj = tf.reduce_sum(w_alpha2 * tf.log(1 + fr / ag), axis=1)
cost2 = -tf.reduce_mean(obj, axis=0)

train_step = tf.train.AdamOptimizer(1e-3).minimize(cost2)


# %%
def get_batch(H, features, labels, num_train, batch_size, L, is_test=False):
    if (is_test):
        idx = np.array(range(batch_size))
    else:
        idx = np.random.randint(num_train, size=batch_size)
    a0 = H[idx, :, :]
    a = np.reshape(features[0][idx, :], (batch_size, L, 1))
    b = np.reshape(features[1][idx, :, :], (batch_size, L, L, 1))
    c = np.reshape(features[2][idx, :, :], (batch_size, L, L, 1))
    d = np.reshape(features[3][idx, :, :], (batch_size, L, L, 1))
    f = np.reshape(features[4][idx, :], (batch_size, L, 1))
    e = np.reshape(labels[idx, :], (batch_size, L, 1))

    return a, b, c, d, e, f, a0


def IC_sum_rate(H, alpha, p, var_noise):
    H = np.square(H)
    fr = np.diag(H) * p
    ag = np.dot(H, p) + var_noise - fr
    y = np.sum(alpha * np.log(1 + fr / ag))
    return y


def np_sum_rate(X, Y, alpha):
    avg = 0
    n = X.shape[0]
    for i in range(n):
        avg += IC_sum_rate(X[i, :, :], alpha[i, :], Y[i, :], 1) / n
    return avg


test_fea = extract_features(X, num_test, K, A)
di_t, intert_t, interf_t, diag_t, labels_t, alpha_t, HHH_t = get_batch(X, test_fea, Y, num_test, num_test, K,
                                                                       is_test=True)
print(labels_t.shape)
# %%
print(np_sum_rate(X, Y, A))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
batch_size = 50
total = num_H * 1000
for ii in range(total // batch_size):

    di, intert, interf, diag, batch_ys, alpha_tr, HHH = get_batch(Xtrain, features, Ytrain, num_H, batch_size, K)
    start_state = np.ones((batch_size, L, 1))
    sess.run(train_step,
             feed_dict={Htrain: HHH, Xinterf: interf, Xintert: intert, Xdiag: di, Xdiag_o: diag, y_: batch_ys,
                        intensity: start_state, w_alpha: alpha_tr})
    if (ii % 10000 == 0):
        start_state = np.ones((num_test, L, 1))
        cost_value = sess.run(cost2, feed_dict={Htrain: HHH_t, Xinterf: interf_t, Xintert: intert_t, Xdiag: di_t,
                                                Xdiag_o: diag_t, y_: labels_t, intensity: start_state,
                                                w_alpha: alpha_t})
        print(cost_value)
    # %%
start_state = np.ones((num_test, L, 1))
cost_value = sess.run(pred2,
                      feed_dict={Htrain: HHH_t, Xinterf: interf_t, Xintert: intert_t, Xdiag: di_t, Xdiag_o: diag_t,
                                 y_: labels_t, intensity: start_state, w_alpha: alpha_t})
pred = np.reshape(cost_value, (500, K))

print('sum rate for IGCNet', np_sum_rate(X, pred, A) * np.log2(np.exp(1)))
print('sum rate for WMMSE', np_sum_rate(X, Y, A) * np.log2(np.exp(1)))
# %%
