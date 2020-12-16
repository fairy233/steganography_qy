import matplotlib.pyplot as plt
# 画sum_loss曲线
f = open("../training/trainingLogs/train_16_log.txt")  # 返回一个文件对象
line = f.readline()  # 调用文件的 readline()方法
train_epoch = []
valid_epoch = []
train_loss = []
valid_loss = []


def getLoss(f, mode):
    line_1 = f.readline()
    line_1_val = line_1.split(' ')
    line_1_val_temp = line_1_val[1].split('/')
    line_2 = f.readline()
    line_3 = f.readline()
    line_3_val = line_3.split('=')
    if mode == 'train':
        train_epoch.append(int(line_1_val_temp[0]))
        train_loss.append(float(line_3_val[3]))  # 画sum_loss曲线
    else:
        valid_epoch.append(int(line_1_val_temp[0]))
        valid_loss.append(float(line_3_val[3]))  # 画sum_loss曲线


while line:   
    # print line             # 后面跟 ',' 将忽略换行符   
    # print(line, end = '')# 在 Python 3 中使用   
    line = f.readline()
    if (line == 'train:\n'):
        getLoss(f, 'train')
    if (line == 'valid:\n'):
        getLoss(f, 'valid')

# print(train_loss)
plt.plot(train_epoch, train_loss, label=u'train_loss')
plt.plot(valid_epoch, valid_loss, label=u'valid_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Sum_loss')
plt.legend()
plt.savefig('../training/trainingLogs/sum_loss_16_1215.jpg')
plt.show()

f.close()