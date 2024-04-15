import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
# from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import pickle

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


#####构造的训练集####
def one_hot(x, char_to_int, alphabet):
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in x]
    # print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    getmatrix = np.array(onehot_encoded)
    return getmatrix


df = pd.read_csv(r'real.csv')
# define input string
# define universe of possible input values
alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

features = []
labels = []
for index, row in df.iterrows():
    mat = [list(one_hot(row['AA'], char_to_int, alphabet))]
    features.append(mat)
    labels.append(np.array([0, 1]) if row['Y'] else np.array([1, 0]))
print(len(features))
# train_idx = random.sample(range(len(features)),k=int(len(features)*0.8))
# test_idx = [x for x in range(len(features)) if x not in train_idx]
with open('train_idx.pkl', 'rb') as f:
    train_idx = pickle.load(f)
with open('test_idx.pkl', 'rb') as f:
    test_idx = pickle.load(f)


out_train_features = []
out_test_features = []
out_train_labels = []
out_test_labels = []
for i in train_idx:
    out_train_features.append(features[i])
    out_train_labels.append(labels[i])
for i in test_idx:
    out_test_features.append(features[i])
    out_test_labels.append(labels[i])

out_train_features = torch.tensor(out_train_features, dtype=torch.float)
out_test_features = torch.tensor(out_test_features, dtype=torch.float)
out_test_labels = torch.tensor(out_test_labels, dtype=torch.float)
out_train_labels = torch.tensor(out_train_labels, dtype=torch.float)
print(1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=5):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

######网络结构##########
class CNN_SEAttention_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*21*21（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 32, 3,stride=1,padding=2) # 输入通道数1，输出通道数10，核的大小5
        # self.se1 = SELayer(32)
        # self.conv2 = nn.Conv2d(10, 20, 3,stride=1,padding=2) # 输入通道数10，输出通道数20，核的大小3
        # self.se2 = SELayer(20)
        # LSTM
        self.lstm = nn.LSTM(input_size=32*11*11, hidden_size=512, num_layers=2)  # 输入 (1, batchsize, 20*12*12)

        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(512, 288) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(288, 2) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*21*21 -> batch*10*21*21（21x21的图像经过一次核为5x5的卷积，输出变为21x21）
        out = F.relu(out) # batch*10*21*21（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*21*21 -> batch*10*10*10（2*2的池化层会减半）
        # out = self.se1(out)
        # out = self.conv2(out) # batch*10*10*10 -> batch*20*12*12（再卷积一次，核的大小是3）
        # out = F.relu(out) # batch*20*12*12
        # out = self.se2(out)
        out = out.view(1, in_size, -1) # batch*20*12*12 -> 1*batch*2880（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out, (_, _) = self.lstm(out)
        out = out.view(in_size, -1)
        out = self.fc1(out) # batch*2880 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*1
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out




# cnn = CNN()
# print(cnn)
# # device="cuda" if torch.cuda.is_available() else "cpu"
# # print("using{}".format(device))
# # cnn.to(device)
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     cnn = cnn.to(device)
# print(cnn)
# print(next(cnn.parameters()).device)


# summary(cnn,input_size=(1,21,21),device=device.type)



##########定义dataset##########
class MyDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels


    def __getitem__(self, item):

        return self.features[item], self.labels[item]


    def __len__(self):
        length = len(self.features)
        return length


########k折划分############
def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs=10, learning_rate=0.0001, weight_decay=0.00001, batch_size=32):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    best_valid_acc = 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
        # net = CNN().to(device)  ### 实例化模型
        # net = CNN_LSTM().to(device)  ### 实例化模型
        # net = CNN_SEAttention().to(device)  ### 实例化模型
        net = CNN_SEAttention_LSTM().to(device)  ### 实例化模型
        ### 每份数据进行训练,体现步骤三####
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        if valid_ls[-1][1] > best_valid_acc:
            torch.save(net.state_dict(), "best_model_cnn_lstm.pth")
            best_valid_acc = valid_ls[-1][1]

        print('*' * 25, '第', i + 1, '折', '*' * 25)
        tr_loss_lst = [l[0] for l in train_ls]
        tr_acc_lst = [l[1] for l in train_ls]
        val_loss_lst = [l[0] for l in valid_ls]
        val_acc_lst = [l[1] for l in valid_ls]
        dic = {'train_loss': tr_loss_lst, 'train_acc': tr_acc_lst,
               'val_loss': val_loss_lst, 'val_acc': val_acc_lst}
        df = pd.DataFrame(dic)
        df.to_excel('第{}折结果数据cal.xlsx'.format(i + 1), index=False)
        print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.4f\n' % train_ls[-1][1], \
              'valid loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    ####体现步骤四#####
    print('train_loss_sum:%.4f' % (train_loss_sum / k), 'train_acc_sum:%.4f\n' % (train_acc_sum / k), \
          'valid_loss_sum:%.4f' % (valid_loss_sum / k), 'valid_acc_sum:%.4f' % (valid_acc_sum / k))


#########训练函数##########
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []  ##存储train_loss,test_loss
    dataset = MyDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    ### 将数据封装成 Dataloder 对应步骤（2）

    # Adam优化算法 or SGD
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (X, y) in enumerate(train_iter):  ###分批训练
            X = torch.as_tensor(X.to(device))
            y = torch.as_tensor(y.to(device))
            output = net(X)
            loss = loss_func(output, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### 得到每个epoch的 loss 和 accuracy
        print("epoch  {}, Loss  {}".format(epoch + 1, total_loss / (i + 1)))
        train_ls.append(log_rmse(0, net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(1, net, test_features, test_labels))
    # print(train_ls,test_ls)
    return train_ls, test_ls


def log_rmse(flag, net, x, y):
    if flag == 1:  ### valid 数据集
        net.eval()
    x = torch.as_tensor(x.to(device))
    y = torch.as_tensor(y.to(device))
    output = net(x)
    # tmp = torch.max(output, 1)
    # tmp = torch.max(output, 1)[1] # 5886
    result = torch.max(output, 1)[1]
    corrects = (result.data == torch.max(y, 1)[1].data).sum().item()
    accuracy = corrects * 100.0 / len(y)  #### 5 是 batch_size
    loss = loss_func(output, y)
    net.train()

    return (loss.data.item(), accuracy)


device = "cuda" if torch.cuda.is_available() else "cpu"
# k 折交叉验证  当只需要绘制测试集的roc曲线时，把下面两行注释会加快绘制速度
# loss_func = nn.BCEWithLogitsLoss()
# k_fold(10, out_train_features, out_train_labels)  ### k=10,十折交叉验证

# roc曲线绘制
# model = CNN().to(device)
# model = CNN_LSTM().to(device)  ### 实例化模型
# model = CNN_SEAttention().to(device)  ### 实例化模型
model = CNN_SEAttention_LSTM().to(device)  ### 实例化模型
model_path = "best_model_cnn_lstm.pth"
model_dict = model.load_state_dict(torch.load(model_path))

model.eval()
test_dataset = MyDataset(out_test_features, out_test_labels)
test_loader = DataLoader(test_dataset, 1, shuffle=False)
total_loss = 0.
correct = 0.
num_classes = 2
with torch.no_grad():
    features, labels = out_test_features.to(device), out_test_labels.to(device)
    output = model(features)
    scores = torch.softmax(output, dim=1).cpu().numpy()
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(out_test_labels[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(out_test_labels.cpu().numpy().ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    ###pr###
    precision_dict = {}
    recall_dict = {}
    average_precision = {}
    for i in range(num_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(out_test_labels[:, i], scores[:, i])
        average_precision[i] = average_precision_score(out_test_labels[:, i], scores[:, i])
        print(precision_dict[i].shape, recall_dict[i].shape, average_precision[i])

    # micro
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(out_test_labels.cpu().numpy().ravel(), scores.ravel())
    average_precision["micro"] = average_precision_score(out_test_labels.cpu().numpy(), scores, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


# plt.style.use('ggplot')
#     # 绘制所有类别平均的pr曲线
# plt.figure(figsize=(8, 8))
# plt.plot(recall_dict['micro'], precision_dict['micro'],label = 'deepO-GlcNac AP={0:0.2f}'.format(average_precision["micro"]))
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.grid(True)
# # plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
# plt.legend(loc="lower right")
# plt.savefig("pred sever pr.jpg")
#
#
# plt.figure(figsize=(8, 8))
# # plt.plot(fpr["micro"], tpr["micro"],
# #          label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
# #          color='deeppink', linestyle=':', linewidth=4)
# #
# # plt.plot(fpr["macro"], tpr["macro"],
# #          label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
# #          color='navy', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"])
#          )
#
# # for i in range(num_classes):
# #     plt.plot(fpr[i], tpr[i], lw=2,
# #              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.grid()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Two-class ROC1')
# plt.legend(loc="lower right")
# plt.savefig('Two-class-ROC-macro.jpg', bbox_inches='tight')
# plt.show()

