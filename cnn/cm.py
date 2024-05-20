import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
# from torchsummary import summary
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.utils.multiclass import unique_labels
from math import sqrt


seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



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
train_idx = random.sample(range(len(features)),k=int(len(features)*0.8))
test_idx = [x for x in range(len(features)) if x not in train_idx]

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




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Conv2d(1, 10, 5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(10, 20, 3,stride=1,padding=2)
        
        self.fc1 = nn.Linear(20*12*12, 288) 
        self.fc2 = nn.Linear(288, 2) 
    def forward(self,x):
        in_size = x.size(0) 
        out = self.conv1(x) 
        out = F.relu(out) 
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out) 
        out = F.relu(out) 
        out = out.view(in_size, -1) 
        out = self.fc1(out) # batch*2880 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*1
        # out = F.log_softmax(out, dim=1) 
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




class MyDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels


    def __getitem__(self, item):

        return self.features[item], self.labels[item]


    def __len__(self):
        length = len(self.features)
        return length



def get_k_fold_data(k, i, X, y): 
 
    assert k > 1
    fold_size = X.shape[0] 

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
     
        X_part, y_part = X[idx, :], y[idx]
        if j == i: 
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs=30, learning_rate=0.0001, weight_decay=0.1, batch_size=5):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    best_valid_acc = 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train) 
        net = CNN().to(device) 
        
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        if valid_ls[-1][1] > best_valid_acc:
            torch.save(net.state_dict(), "best_model_real.pth")
            best_valid_acc = valid_ls[-1][1]

        print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.4f\n' % train_ls[-1][1], \
              'valid loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
   
    print('train_loss_sum:%.4f' % (train_loss_sum / k), 'train_acc_sum:%.4f\n' % (train_acc_sum / k), \
          'valid_loss_sum:%.4f' % (valid_loss_sum / k), 'valid_acc_sum:%.4f' % (valid_acc_sum / k))



def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []  ##存储train_loss,test_loss
    dataset = MyDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)


  
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(num_epochs):
        for X, y in train_iter:  ###分批训练
            X = torch.as_tensor(X.to(device))
            y = torch.as_tensor(y.to(device))
            output = net(X)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
     
        train_ls.append(log_rmse(0, net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(1, net, test_features, test_labels))
    # print(train_ls,test_ls)
    return train_ls, test_ls


def log_rmse(flag, net, x, y):
    if flag == 1:  
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def calculate_MCC(TP,FP,FN,TN):
    numerator = (TP * TN) - (FP * FN) 
    denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) 
    result = numerator/denominator
    return result

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_func = nn.BCEWithLogitsLoss()
k_fold(10, out_train_features, out_train_labels)  


model = CNN().to(device)
model_path = r'..\best_model_real0.92.pth'
model_dict=model.load_state_dict(torch.load(model_path))

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
    scores_cla = scores.argmax(1)
    y_pred = scores_cla.tolist()
    # print(y_pred)
    # print(type(y_pred))
    y_ture = torch.max(labels, 1)[1].cpu().numpy()
    y_ture = y_ture.tolist()
    # print(y_ture)
    # print(type(y_ture))
    # print(len(y_ture))
    # print(len(y_pred))


    ###confusion_matrix####
    cm1 = confusion_matrix(y_ture,y_pred)
    # print(cm1)
    # sns.heatmap(cm1, annot=True)
    # plt.show()
    class_names = np.array(["0", "1"])
    plot_confusion_matrix(y_ture, y_pred, classes=class_names, normalize=False)
    # FP = cm1.sum(axis=0) - np.diag(cm1)
    # FN = cm1.sum(axis=1) - np.diag(cm1)
    # TP = np.diag(cm1)
    # TN = cm1.sum() - (FP + FN + TP)
    TP = cm1[0][0]
    FP = cm1[0][1]
    FN = cm1[1][0]
    TN = cm1[1][1]
    # print('FP:{0:0.2f}'.format(FP))
    # print('FN:{0:0.2f}'.format(FN))
    # print('TP:{0:0.2f}'.format(TP))
    # print('TN:{0:0.2f}'.format(TN))
    Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
    Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
    Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
    Acc = round((TP+TN)/(TP+FN+TN+FP), 3) if TP+FN+TN+FP != 0 else 0.
    print('Precixion:{0:0.2f}'.format(Precision))
    print('Recall:{0:0.2f}'.format(Recall))
    print('Specificity:{0:0.2f}'.format(Specificity))
    MCC = calculate_MCC(TP, FP, FN, TN)
    print('MCC:{0:0.2f}'.format(MCC))
    print('Acc:{0:0.2f}'.format(Acc))

    #acc
    result = torch.max(output, 1)[1]
    corrects = (result.data == torch.max(labels, 1)[1].data).sum().item()
    accuracy = corrects * 100.0 / len(labels)  #### 5 是 batch_size
    print('test accuracy:{0:0.2f}'.format(accuracy))


