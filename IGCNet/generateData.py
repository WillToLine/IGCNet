import numpy as np
import matplotlib.pyplot as plt
import function_wmmse_powercontrol as wf
import time
import torch


def generate_wGaussian(K, num_H, var_noise=1.0, Pmin=0, seed=2017):
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


def extract_labels(y, n, L):
    labels = np.zeros((n, L, 2))
    for ii in range(n):
        for jj in range(L):
            if (abs(y[ii][jj]) < 1e-4):
                labels[ii, jj, :] = [1, 0]
            else:
                labels[ii, jj, :] = [0, 1]
    return labels


def getRawBatch(H, features, labels, num_train, batch_size, L, is_test=False):
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

    a = torch.tensor(a, dtype=torch.float32)
    f = torch.tensor(f, dtype=torch.float32)
    a0 = torch.tensor(a0, dtype=torch.float32)

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


def getBatch(interf, intert, diag, intense, w):
    interf = torch.tensor(interf, dtype=torch.float32)
    intert = torch.tensor(intert, dtype=torch.float32)
    diag = torch.tensor(diag, dtype=torch.float32)
    # intense = torch.tensor(start_state, dtype=torch.float32)
    # w = torch.tensor(alpha_tr, dtype=torch.float32)

    ones_tensor = torch.ones((1, K), dtype=torch.float32)
    intense2 = torch.matmul(intense, ones_tensor)
    intense2 = intense2.transpose(1, 2).reshape((-1, K, K, 1))

    w2 = torch.matmul(w, ones_tensor)
    w2 = w2.transpose(1, 2).reshape((-1, K, K, 1))

    xBatch = torch.cat((interf, intert, diag, intense2, w2), dim=3)
    xBatch = xBatch.permute(0, 3, 1, 2)
    return xBatch


def transposeX(di, start_state, alpha_tr):
    d = di.clone().permute(0, 2, 1)
    ss = start_state.clone().permute(0, 2, 1)
    at = alpha_tr.clone().permute(0, 2, 1)
    return d, ss, at



K = 20  # number of users
num_H = 2000  # number of training samples
num_test = 500  # number of testing  samples
training_epochs = 50  # number of training epochs
trainseed = 0  # set random seed for training set
testseed = 7  # set random seed for test set
batch_size = 50

print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n' % (K, num_H, training_epochs))
var_db = 0
var = 1 / 10 ** (var_db / 10)

# Xtrain, Ytrain, Atrain, wtime = generate_wGaussian(K, num_H, seed=trainseed, var_noise=var)
# X, Y, A, wmmsetime = generate_wGaussian(K, num_test, seed=testseed, var_noise=var)
# Xtrain = Xtrain.transpose()
# X = X.transpose()
# Ytrain = Ytrain.transpose()
# Y = Y.transpose()
#
# print(Xtrain.shape, Ytrain.shape)
#
# Xtrain = Xtrain.reshape((-1, K, K))
# X = X.reshape((-1, K, K))
#
# features = extract_features(Xtrain, num_H, K, Atrain)
# labels = extract_labels(Ytrain, num_H, K)
# labels_t = extract_labels(Y, num_test, K)
#
# test_fea = extract_features(X, num_test, K, A)
# di_t, intert_t, interf_t, diag_t, labels_t, alpha_t, HHH_t = getRawBatch(X, test_fea, Y, num_test, num_test, K,
#                                                                        is_test=True)
# print(labels_t.shape)
#
# di, intert, interf, diag, batch_ys, alpha_tr, HHH = getRawBatch(Xtrain, features, Ytrain, num_H, batch_size, K)
# start_state = torch.ones((batch_size, K, 1), dtype=torch.float32)
#
# print("interf", interf.shape)  # Xinterf
# print("intert", intert.shape)  # Xintert
# print("diag", diag.shape)  # Xdiag_o
# print("start_state", start_state.shape)  # intensity
# print("alpha_tr", alpha_tr.shape)  # w_alpha
# print("di", di.shape)
#
# xtuple = transposeX(di, start_state, alpha_tr)
# print(xtuple[0].shape)
# print(xtuple[1].shape)
# print(xtuple[2].shape)
#
# x = getBatch(interf, intert, diag, start_state, alpha_tr)
# print("x", x.shape)



