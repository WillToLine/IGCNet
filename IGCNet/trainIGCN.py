import numpy as np  # import numpy
import matplotlib.pyplot as plt  # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol as wf
from IGCN import IGCNet
from generateData import *
import torch
import time

lr = 0.001
epoch = 30
total_train_loss = 0.0
total_time = 0.

batch_size = 50
total = num_H * 1000


# loss
def lossFunction(p, Htrain, w_alpha):
    H = torch.square(Htrain)
    signalSum = torch.matmul(H, p).view(-1, K)
    p = p.view(-1, K)
    diagonal_elements = []
    for i in range(H.size(0)):
        # 提取第 i 个二维平面的对角元素
        diagonal_elements.append(torch.diag(H[i]))
    # 将结果拼接成一个张量
    H_diag = torch.stack(diagonal_elements)
    signal = H_diag * p
    interNoise = signalSum - signal + var
    w_alpha2 = w_alpha.reshape(-1, K)

    lossObj = torch.sum(w_alpha2 * torch.log2(1 + signal / interNoise), dim=1)
    cost = -torch.mean(lossObj, dim=0)
    return cost


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
my_model = IGCNet()
my_model.to(device)
# 优化器
optim = torch.optim.Adam(my_model.parameters(), lr=lr)


# ---------------------------------------Generate Data--------------------------------------#
Xtrain, Ytrain, Atrain, wtime = generate_wGaussian(K, num_H, seed=trainseed, var_noise=var)
X, Y, A, wmmsetime = generate_wGaussian(K, num_test, seed=testseed, var_noise=var)
Xtrain = Xtrain.transpose()
X = X.transpose()
Ytrain = Ytrain.transpose()
Y = Y.transpose()

print(Xtrain.shape, Ytrain.shape)

Xtrain = Xtrain.reshape((-1, K, K))
X = X.reshape((-1, K, K))

features = extract_features(Xtrain, num_H, K, Atrain)
test_fea = extract_features(X, num_test, K, A)
# labels = extract_labels(Ytrain, num_H, K)
# labels_t = extract_labels(Y, num_test, K)

# di_t, intert_t, interf_t, diag_t, labels_t, alpha_t, HHH_t = getRawBatch(X, test_fea, Y, num_test, num_test, K,
#                                                                        is_test=True)
# print(labels_t.shape)

# ---------------------------------------Generate Data--------------------------------------#
print(np_sum_rate(X, Y, A))

for i in range(total // batch_size):
    print(f"---------------第{i}轮训练开始-----------------")
    # total_test_loss = 0.0
    # total_accuracy = 0.0
    # train_time = 0.
    # start_time = time.time()
    # my_model.train()
    for ii in range(num_H // batch_size):
        di, intert, interf, diag, batch_ys, alpha_tr, Htrain = getRawBatch(Xtrain, features, Ytrain, num_H, batch_size, K)
        start_state = torch.ones((batch_size, K, 1))
        xBatch = getBatch(interf, intert, diag, start_state, alpha_tr)
        d, ss, at = transposeX(di, start_state, alpha_tr)
        xBatch.to(device)
        d.to(device)
        ss.to(device)
        at.to(device)
        # 梯度归零
        optim.zero_grad()
        output = my_model(xBatch, d, ss, at)
        outLoss = lossFunction(output, Htrain, alpha_tr)
        # 反向梯度传播
        outLoss.backward()
        # 使用优化器更新参数
        optim.step()

    with torch.no_grad():
        di_t, intert_t, interf_t, diag_t, labels_t, alpha_t, HHH_t = getRawBatch(X, test_fea, Y, num_test, num_test, K,
                                                                                 is_test=True)
        start_state = torch.ones((num_test, K, 1))
        xTest = getBatch(interf_t, intert_t, diag_t, start_state, alpha_t)
        dTest, ssTest, atTest = transposeX(di_t, start_state, alpha_t)
        xTest.to(device)
        dTest.to(device)
        ssTest.to(device)
        atTest.to(device)

        test_out = my_model(xTest, dTest, ssTest, atTest)
        outLoss = lossFunction(test_out, HHH_t, alpha_t)
        print("cost:", outLoss)

    torch.save(my_model.state_dict(), "./save_model/cpu_{}.pth".format(i))

