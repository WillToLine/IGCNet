from IGCN import IGCNet, init_weights
from IGCNet.utils.generateData import *
import torch

lr = 0.001
epoch = 30
total_train_loss = 0.0
total_time = 0.
batch_size = 50


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

    costObj = torch.sum(w_alpha2 * torch.log2(1 + signal / interNoise), dim=1)
    loss = -torch.mean(costObj, dim=0)
    return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
my_model = IGCNet()
my_model.apply(init_weights)
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

print(X.shape, Y.shape)

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
total_cost = []
total_loss = []
sum_rate = []

for i in range(epoch):
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
        xBatch = xBatch.to(device)
        d = d.to(device)
        ss = ss.to(device)
        at = at.to(device)
        Htrain = Htrain.to(device)
        alpha_tr = alpha_tr.to(device)
        # 梯度归零
        optim.zero_grad()
        output = my_model(xBatch, d, ss, at)
        outLoss = lossFunction(output, Htrain, alpha_tr)
        # 反向梯度传播
        outLoss.backward()
        # 使用优化器更新参数
        optim.step()

    # my_model.eval()
    with torch.no_grad():
        di_t, intert_t, interf_t, diag_t, labels_t, alpha_t, HHH_t = getRawBatch(X, test_fea, Y, num_test, num_test, K,
                                                                                 is_test=True)
        start_state = torch.ones((num_test, K, 1))
        xTest = getBatch(interf_t, intert_t, diag_t, start_state, alpha_t)
        dTest, ssTest, atTest = transposeX(di_t, start_state, alpha_t)
        xTest = xTest.to(device)
        dTest = dTest.to(device)
        ssTest = ssTest.to(device)
        atTest = atTest.to(device)
        HHH_t = HHH_t.to(device)
        alpha_t = alpha_t.to(device)

        test_out = my_model(xTest, dTest, ssTest, atTest)
        outLoss = lossFunction(test_out, HHH_t, alpha_t)
        print("loss:", outLoss)
        total_loss.append(outLoss.item())
        total_cost.append(-outLoss.item())
        if i == epoch - 1:
            pred = test_out.cpu().data.numpy()
            pred = np.reshape(pred, (num_test, K))

    torch.save(my_model.state_dict(), "./save_model/cpu_{}.pth".format(i))


rateIGCN = np_sum_rate(X, pred, A) * np.log2(np.exp(1))
rateWMMSE = np_sum_rate(X, Y, A) * np.log2(np.exp(1))
print("Sum rate of IGCN", rateIGCN)
print("Sum rate of WMMSE", rateWMMSE)

sum_rate = [rateWMMSE] * len(total_loss)
x = [i for i in range(epoch)]
plt.figure(1)
plt.plot(x, sum_rate, label="WMMSE")
plt.plot(x, total_cost, label="IGCN Cost")
plt.plot(x, total_loss, label="IGCN Loss")
plt.xlabel("Epoch")
plt.ylabel("Cost / Loss")
plt.title("Loss and Cost of IGCN")
plt.legend()
plt.show()



