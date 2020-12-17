import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.loss import *
from utils.AverageMeter import AverageMeter
from models.Hnet_base4 import HNet
from models.Rnet_base import RNet
from utils.util import *


# only train !
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=20, help='Batch size.'
    )
    parser.add_argument(
        '--image_size', type=int, default=256, help='image size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of data loading workers.',
    )
    parser.add_argument(
        '--images_path', type=str, default='/media/a3080/b696e7b7-1f6d-4f3e-967e-d164ff107a68/qiao/HDR1', help='Path to coverImage data.'
    )
    parser.add_argument(
        '--use_gpu', type=bool, default=True, help='Use GPU for training.'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate, default=0.001'
    )
    parser.add_argument(
        '--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5'
    )
    parser.add_argument(
        '--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')

    parser.add_argument(
        '--beta', type=float, default=0.75, help='hyper parameter of loss_sum'
    )
    parser.add_argument(
        '--epochs', type=int, default=800, help='the num of training times'
    )
    parser.add_argument(
        '--checkpoint_freq', type=int, default=100, help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '--loss_freq', type=int, default=10, help='Report (average) loss every x epochs.',
    )
    parser.add_argument(
        '--result_freq', type=int, default=20, help='save the resultPictures every x epochs'
    )
    parser.add_argument(
        '--checkpoint_path', default='./training', help='Path for checkpointing.',
    )
    parser.add_argument(
        '--log_path', default='./training', help='log path'
    )
    parser.add_argument(
        '--result_pics', default='./training', help='folder to output training images'
    )
    parser.add_argument(
        # './training1125/checkPoints/H_epoch0150_sumloss=0.001211.pth'
        '--Hnet', default='', help="path to Hidenet (to continue training)"
    )
    parser.add_argument(
        # './training1125/checkPoints/R_epoch0150_sumloss=0.001211.pth'
        '--Rnet', default='', help="path to Revealnet (to continue training)"
    )

    return parser.parse_args()


def train(data_loader, epoch, Hnet, Rnet):
    start_time = time.time()
    train_Hlosses = AverageMeter()
    train_Rlosses = AverageMeter()
    train_SumLosses = AverageMeter()

    # 早停法
    # patience = 7
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    Hnet.train()  # 训练
    Rnet.train()

    # batch_size循环
    for i, data in enumerate(data_loader):
        Hnet.zero_grad()
        Rnet.zero_grad()

        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step

        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]
        # concat cover images and secret images as input of H-net
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        if opt.use_gpu:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img.clone(), requires_grad=False)
        cover_imgv = Variable(cover_img.clone(), requires_grad=False)
        secret_imgv = Variable(secret_img.clone(), requires_grad=False)

        stego = Hnet(concat_imgv)
        errH = loss(stego, cover_imgv)  # loss between cover and container
        train_Hlosses.update(errH.item(), this_batch_size)

        secret_rev = Rnet(stego)
        errR = loss(secret_rev, secret_imgv)  # loss between secret and revealed secret
        train_Rlosses.update(errR.item(), this_batch_size)

        err_sum = errH + opt.beta * errR   # sum_loss
        train_SumLosses.update(err_sum.item(), this_batch_size)

        err_sum.backward()
        optimizerH.step()
        optimizerR.step()

    # TODO: early stop
    # early_stopping(val_SumLosses.avg, Hnet)
    # early_stopping(val_SumLosses.avg, Rnet)

    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

    # save result pictures
    if (epoch % opt.result_freq == 0) or (epoch == opt.epochs - 1):
        save_pic('train', cover_img, stego, secret_img, secret_rev, opt.result_pics, opt.batch_size, epoch)
        # save_batch_pic('train', cover_img, stego, secret_img, secret_rev, opt.result_pics, opt.batch_size, epoch)

    # save model params
    if epoch % opt.checkpoint_freq == 0 or epoch == opt.epochs - 1:
        torch.save(
            Hnet.state_dict(),
            os.path.join(opt.checkpoint_path,
                         'H_epoch%04d_sumloss%.6f_lr%.6f.pth' % (epoch, train_SumLosses.avg, optimizerH.param_groups[0]['lr']))
        )
        torch.save(
            Rnet.state_dict(),
            os.path.join(opt.checkpoint_path,
                         'R_epoch%04d_sumloss%.6f_lr%.6f.pth' % (epoch, train_SumLosses.avg, optimizerR.param_groups[0]['lr']))
        )

    # print log
    epoch_time = time.time() - start_time
    epoch_log = 'train:' + '\n'
    epoch_log += "epoch %d/%d : " % (epoch, opt.epochs)
    epoch_log += "one epoch time is %.0fm %.0fs" % (epoch_time // 60, epoch_time % 60) + "\n"
    epoch_log += "learning rate: optimizerH_lr = %.8f\t optimizerR_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
    # schedulerH.get_lr()[0] schedulerR.get_lr()[0]
    epoch_log += "Hloss=%.6f\t Rloss=%.6f\t sumLoss=%.6f" % (
    train_Hlosses.avg, train_Rlosses.avg, train_SumLosses.avg) + "\n"

    if epoch % opt.loss_freq == 0:
        print_log(epoch_log, logPath)
    else:
        print_log(epoch_log, logPath, console=False)


def main():
    # 定义全局参数
    global opt, logPath, optimizerH, optimizerR, schedulerH, schedulerR
    opt = parse_args()

    if torch.cuda.is_available() and opt.use_gpu:
        print("CUDA is available!")

    # 创建文件夹
    try:
        opt.checkpoint_path += "/checkPoints"
        opt.result_pics += "/resultPics"
        opt.log_path += "/trainingLogs"

        if not os.path.exists(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        if not os.path.exists(opt.result_pics):
            os.makedirs(opt.result_pics)
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
    except OSError:
        print("mkdir failed!")

    # 训练的log
    logPath = opt.log_path + '/train_%d_log.txt' % opt.batch_size

    print_log(time.asctime(time.localtime(time.time())), logPath, False)
    print_log(str(opt), logPath, False)

    # 准备数据集，train/valid
    print_log('prepare train dataset', logPath)
    data_dir = opt.images_path
    train_dataset = DirectoryDataset(os.path.join(data_dir, 'train'), preprocess=transforms)
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    assert dataloader
    print_log('train dataset has been prepared!', logPath)

    # 初始化模型 Hnet Rnet
    modelH = HNet()
    modelR = RNet()

    if opt.use_gpu:
        modelH.cuda()
        modelR.cuda()
        torch.backends.cudnn.benchmark = True

    # TODO: 参数初始化
    # modelH.apply(weights_init)
    # modelR.apply(weights_init)

    # whether to load pre-trained model
    if opt.Hnet != '':
        modelH.load_state_dict(torch.load(opt.Hnet))
        print_network(modelH, logPath)
    if opt.Rnet != '':
        modelR.load_state_dict(torch.load(opt.Rnet))
        print_network(modelR, logPath)

    # 优化器  beta1=0.9, beta2=0.999
    optimizerH = torch.optim.Adam(modelH.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    # 训练策略，学习率下降, 若训练集的loss值一直不变，就调小学习率
    # schedulerH = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=10, verbose=True, min_lr=0.0000001)
    optimizerR = torch.optim.Adam(modelR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    # schedulerR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=10, verbose=True, min_lr=0.0000001)
    schedulerH = torch.optim.lr_scheduler.ExponentialLR(optimizerH, gamma=0.8)
    schedulerR = torch.optim.lr_scheduler.ExponentialLR(optimizerR, gamma=0.8)
    print_log("training is beginning ......................................", logPath)

    for epoch in range(opt.epochs):
        # 只有训练的时候才会计算和更新梯度
        train(dataloader, epoch, Hnet=modelH, Rnet=modelR)
        # 用验证集的loss来调整学习率
        # schedulerH.step(val_sumloss)
        # schedulerR.step(val_rloss)
        if epoch != 0 and epoch % 200 == 0:
            schedulerH.step()
            schedulerR.step()


if __name__ == '__main__':
    main()