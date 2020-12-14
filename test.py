import argparse, os, time, cv2, torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.util import *
from utils.loss import *
from models.Hnet_base import HNet
from models.Rnet_base import RNet


# test
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_size', type=int, default=256, help='image size'
    )
    parser.add_argument(
        '--beta', type=float, default=0.75, help='hyper parameter of loss_sum'
    )
    parser.add_argument(
        '--test_path', default='./hdr/test', help='test hdr images path'
    )
    parser.add_argument(
        '--test_pics', default='./test/resultPics', help='folder to output test images'
    )
    parser.add_argument(
        '--test_log', default='./test/testLog.txt', help='test log'
    )
    parser.add_argument(
        '--Hnet', default='./training1211/checkPoints/H_epoch0799_sumloss0.000568_lr0.001000.pth', help="path to Hidenet (to continue training)"
    )
    parser.add_argument(
        '--Rnet', default='./training1211/checkPoints/R_epoch0799_sumloss0.000568_lr0.001000.pth', help="path to Revealnet (to continue training)"
    )
    return parser.parse_args()


# 1. 参数
opt = parse_args()
print_log(time.asctime(time.localtime(time.time())), opt.test_log, False)

# 2. 创建输出结果文件夹
try:
    if not os.path.exists(opt.test_pics):
        os.makedirs(opt.test_pics)
except OSError:
    print("mkdir failed!")

# 把所有参数打印在日志中
print_log(str(opt), opt.test_log, False)

if torch.cuda.is_available():
    print("CUDA is available!")


# 3. 准备test数据集
print_log('prepare test dataset', opt.test_log)
test_dataset = DirectoryDataset(opt.test_path, preprocess=transforms)
test_loader = DataLoader(test_dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=True)
assert test_loader
print_log('test dataset has been prepared!', opt.test_log)

# 4. 加载模型并打印
modelH = HNet()
modelR = RNet()
if torch.cuda.is_available():
    modelH.cuda()
    modelR.cuda()
    torch.backends.cudnn.benchmark = True

modelH.load_state_dict(torch.load(opt.Hnet))
print_network(modelH, opt.test_log)
modelR.load_state_dict(torch.load(opt.Rnet))
print_network(modelR, opt.test_log)


def test(data_loader, Hnet, Rnet):
    print_log("---------- test begin ---------", opt.test_log)
    Hnet.eval()
    Rnet.eval()
    for i, data in enumerate(data_loader):
        all_pics = data
        this_batch_size = int(all_pics.size()[0] / 2)

        cover_img = all_pics[0:this_batch_size, :, :, :]
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        if torch.cuda.is_available():
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, requires_grad=False)
        cover_imgv = Variable(cover_img, requires_grad=False)
        secret_imgv = Variable(secret_img, requires_grad=False)

        stego = Hnet(concat_imgv)
        secret_rev = Rnet(stego)

        errH = loss(stego, cover_imgv)  # loss between cover and container
        errR = loss(secret_rev, secret_imgv)  # loss between secret and revealed secret
        err_sum = errH + opt.beta * errR

        save_pic('test', cover_img, stego, secret_img, secret_rev, opt.test_pics, 2, i)
        test_log = '%d: loss is %.6f' % (i, err_sum.item()) + '\n'
        print_log(test_log, opt.test_log)

    print_log("---------- test end ----------", opt.test_log)

# 开始测试！
test(test_loader, Hnet=modelH, Rnet=modelR)


