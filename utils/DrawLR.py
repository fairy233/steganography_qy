import matplotlib.pyplot as plt
# 画train的 LR曲线
# H和R的学习率相同

f = open("../training/trainingLogs/train_16_log.txt")  # 返回一个文件对象
line = f.readline()  # 调用文件的 readline()方法
train_epoch = []
train_lr = []
# train_lr_H = []
# train_lr_R = []

def getLR(f, mode):
    line_1 = f.readline()
    line_1_val = line_1.split(' ')
    line_1_val_temp = line_1_val[1].split('/')
    line_2 = f.readline()
    line_2_val = line_2.split('=')
    line_2_val_val = line_2_val[1].split('\t')
    if mode == 'train':
        train_epoch.append(int(line_1_val_temp[0]))
        train_lr.append(float(line_2_val_val[0]))  # 画lr曲线
        # train_lr_H.append(float(line_2_val_val[0]))  # 画H_lr曲线
        # train_lr_R.append(float(line_2_val[2]))  # 画R_lr曲线


while line:
    line = f.readline()
    if (line == 'train:\n'):
        getLR(f, 'train')
# print(train_lr_H)
# print(train_lr_R)

# plt.plot(train_epoch, train_lr_H, label=u'train_lr_H')
# plt.plot(train_epoch, train_lr_R, label=u'train_lr_R')
plt.plot(train_epoch, train_lr, label=u'train_lr')
plt.xlabel('epochs')
plt.ylabel('lr')
plt.title('train_lr')
plt.legend()
plt.savefig('../training/trainingLogs/lr_16_1216.jpg')
plt.show()

f.close()