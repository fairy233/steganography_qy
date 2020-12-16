import os
import numpy as np
from numpy.random import uniform
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.utils as vutils
import math


# 返回绝对路径
def process_path(directory, create=False):
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    directory = os.path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


# 返回目录名 文件名 扩展名
def split_path(directory):
    directory = process_path(directory)
    name, ext = os.path.splitext(os.path.basename(directory))
    return os.path.dirname(directory), name, ext


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)].astype(np.float32)
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


def torch2cv(t_img):
    t_img = t_img.detach().cpu().numpy()
    t_img = t_img[0, :, :, :]  # qu batch
    # t_img = t_img.squeeze(0)
    return t_img.swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]


def resize(x, size):
    return cv2.resize(x, size)


def hdr2ldr(hdr):
    Du = cv2.createTonemapDurand(2)
    ldr = Du.process(hdr)
    ldr = np.clip(ldr, 0, 1)
    return ldr


# 对输入图像的预处理的函数--图像增强（裁剪和归一化，图像翻转flip, 高斯噪声）
def transforms(hdr):
    hdr_size = np.array(hdr.shape)
    hdr = random_crop(hdr, resize=True)  # hdr 是一个numpy (256,256,3)

    # 添加指数E分量
    # e = get_e_from_float(hdr)
    # hdr = np.concatenate((hdr, e), axis=2)  # hdr 是一个numpy (256,256,4)

    # 随机翻转
    if np.random.rand() < 0.5:
        hdr = cv2.flip(hdr, 1)  # 1 水平翻转 0 垂直翻转 -1 水平垂直翻转
    # 添加高斯随机噪声
    if np.random.rand() < 0.5:
        hdr = random_noise(hdr)

    # 归一化
    hdr = normImage(hdr)
    hdr = cv2torch(hdr)  # 转为(3,256,256) tensor
    return hdr


def testTransforms(hdr):
    hdr = random_crop(hdr, resize=True)  # hdr 是一个numpy (256,256,3)
    # hdr = cv2.resize(hdr, (256, 256))  # 把裁剪后的图resize到256，256范围

    # 归一化
    hdr = normImage(hdr)
    hdr = cv2torch(hdr)  # 转为(3,256,256) tensor
    return hdr


def print_log(log_info, log_path, console=True):
    log_info += '\n'
    if console:
        print(log_info)
    # debug mode will not write logs into files
        # write logs into log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:  # 如果地址下有文件，打开这个文件后，再写入日志
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


def print_network(net, logPath):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath, console=False)
    print_log('Total number of parameters: %d' % num_params, logPath, console=False)


def save_pic(phase, cover, stego, secret, secret_rev, save_path, batch_size, epoch):
    # tensor  --> numpy.narray batch=1
    cover = torch2cv(cover)
    secret = torch2cv(secret)
    stego = torch2cv(stego)
    secret_rev = torch2cv(secret_rev)

    #  np.vstack  np.hstack
    showContainer = np.hstack((cover, stego))
    showReveal = np.hstack((secret, secret_rev))
    resultImg = np.vstack((showContainer, showReveal))

    # tone map  ldr
    cover_ldr = (hdr2ldr(cover) * 255).astype(int)
    secret_ldr = (hdr2ldr(secret) * 255).astype(int)
    stego_ldr = (hdr2ldr(stego) * 255).astype(int)
    secret_rev_ldr = (hdr2ldr(secret_rev) * 255).astype(int)

    showContainer_ldr = np.hstack((cover_ldr, stego_ldr))
    showReveal_ldr = np.hstack((secret_ldr, secret_rev_ldr))
    resultImg_ldr = np.vstack((showContainer_ldr, showReveal_ldr))

    if phase == 'test':
        # diff图像，只在test中保存，且只保存ldr格式
        cover_diff = cover - stego
        secret_diff = secret - secret_rev

        cover_diff_ldr = (hdr2ldr(cover_diff) * 255).astype(int)
        secret_diff_ldr = (hdr2ldr(secret_diff) * 255).astype(int)
        diffImg_ldr = np.vstack((cover_diff_ldr, secret_diff_ldr))
        # 单独保存cover、stego、secret、secret_rev和diff
        cv2.imwrite('%s/cover_%02d.hdr' % (save_path, epoch), cover)
        cv2.imwrite('%s/stego_%02d.hdr' % (save_path, epoch), stego)
        cv2.imwrite('%s/secret_%02d.hdr' % (save_path, epoch), secret)
        cv2.imwrite('%s/secret_rev_%02d.hdr' % (save_path, epoch), secret_rev)
        cv2.imwrite('%s/test_diff_%02d.jpg' % (save_path, epoch), diffImg_ldr)

    elif phase == 'train':
        resultImgName = '%s/trainResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)
        # result hdr
        cv2.imwrite(resultImgName, resultImg)
        # result ldr
        cv2.imwrite(resultImgName + '.jpg', resultImg_ldr)
    else:
        resultImgName = '%s/valResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)
        # result hdr
        cv2.imwrite(resultImgName, resultImg)
        # result ldr
        cv2.imwrite(resultImgName + '.jpg', resultImg_ldr)


def save_batch_pic(phase, cover, stego, secret, secret_rev, save_path, batch_size, epoch):
    # hdr geshi bukeyi !!!!!
    # tensor  --> numpy.narray
    showContainer = torch.cat([cover, stego], 0)
    showReveal = torch.cat([secret, secret_rev], 0)
    resultImg = torch.cat((showContainer, showReveal), 0)

    # tone map  ldr
    # cover_ldr = (hdr2ldr(cover) * 255).astype(int)
    # secret_ldr = (hdr2ldr(secret) * 255).astype(int)
    # stego_ldr = (hdr2ldr(stego) * 255).astype(int)
    # secret_rev_ldr = (hdr2ldr(secret_rev) * 255).astype(int)
    #
    # showContainer_ldr = torch.cat([cover_ldr, stego_ldr], 0)
    # showReveal_ldr = torch.cat([secret_ldr, secret_rev_ldr], 0)
    # resultImg_ldr = torch.cat([showContainer_ldr, showReveal_ldr], 0)

    if phase == 'train':
        resultImgName = '%s/trainResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)
        # result hdr
        vutils.save_image(resultImg, resultImgName, nrow=int(batch_size/2), padding=1, normalize=True)
        # result ldr
        # vutils.save_image(resultImg_ldr, resultImgName + '.jpg', nrow=int(batch_size/2), padding=1, normalize=True)


    else:
        resultImgName = '%s/valResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)
        # result hdr
        vutils.save_image(resultImg, resultImgName, nrow=int(batch_size/2), padding=1, normalize=True)
        # result ldr
        # vutils.save_image(resultImg_ldr, resultImgName + '.jpg', nrow=int(batch_size/2), padding=1, normalize=True)


def random_noise(img, im_noise=[0.0, 0.0001]):
    # img = img.astype(np.float32)
    # 三个参数，均值、方差和输出的size 均值0 方差为0.0001
    noise = np.random.normal(im_noise[0], im_noise[1], img.shape).astype(np.float32)
    noise[noise > 0.1] = 0.1
    noise[noise < -0.1] = -0.1  # 保证噪声在[-0.1,0.1]范围
    img = img + noise
    # img[img < 1e-5] = 1e-5
    # img[img > 1] = 1
    return img


# 随机裁剪
def random_crop(img, sub_im_sc=[6, 6], resize=False, rez_im_sc=[256, 256]):
    sub_im_sc = np.array(sub_im_sc)
    img_size = np.array(img.shape)  # get size [256,256,3]

    if sum(img_size[:2] < 256) >= 1:  # img_size[:2] [256,256]
        print('img size error(too small)!')
        raise IndexError

    # np.random.rand(2) 生成两个服从均匀分布的随机数（0-1） 最终是2-6 * 128  256-768
    # crop_size 是随机裁剪的宽高。可能不是正方形
    crop_size = (2 + (sub_im_sc - 2) * np.random.rand(2)).astype(np.int) * 128
    # print('crop_size: ', crop_size)
    # print('img_size: ', img_size)
    crop_size[crop_size >= img_size[:2]] = 256
    h, w = crop_size
    # print(h,w)
    h_start, w_start = (np.random.rand(2) * (img_size[:2] - [h, w])).astype(np.int)
    # print(h_start,w_start)

    img_crop = img[h_start:h_start + h, w_start:w_start + w, :]  # 这个剪裁之后的图片大小不一定是256*256， 是有个范围的

    if resize == True:
        img_crop = cv2.resize(img_crop, (rez_im_sc[0], rez_im_sc[1]))  # 把裁剪后的图resize到256，256范围

    return img_crop


def get_e_from_float(RGB, normalize='minmax', esp=1e-5):
    if RGB.ndim == 3:
        zeros = np.sum(RGB, axis=2) == 0
        max_value = np.max(RGB, axis=2)
        e = np.floor(np.log2(max_value)) + 129
        e[zeros] = 0
        if normalize == 'minmax':
            e_min = np.min(e)
            e_max = np.max(e)
            e = (e - e_min) / (e_max - e_min + esp)
        elif normalize == 'log':
            e_min = np.min(e)
            e_max = np.max(e)
            e = np.log(e + 1 - e_min) / np.log(e_max + 1 - e_min + esp)
        else:
            raise NotImplementedError
        e = np.expand_dims(e, axis=2)
        return e
    else:
        raise NotImplementedError


def normImage(hdr, normalize='minmax', esp=1e-5):
    minvalue = np.min(hdr)
    maxvalue = np.max(hdr)
    if normalize == 'minmax':
        hdr = (hdr - minvalue) / (maxvalue - minvalue + esp)
    elif normalize == 'log':
        hdr = np.log(hdr + 1 - minvalue) / (np.log(maxvalue + 1 - minvalue)  + esp)
    else:
        raise NotImplementedError
    return hdr


class DirectoryDataset(Dataset):
    def __init__(
            self,
            image_path='hdr/train',
            data_extensions=['.hdr', '.exr', '.jpeg'],
            preprocess=None,
    ):
        super(DirectoryDataset, self).__init__()

        image_path = process_path(image_path)
        self.image_list = []

        for root, _, fnames in sorted(os.walk(image_path)):
            for fname in fnames:
                if any(
                        fname.lower().endswith(extension)
                        for extension in data_extensions
                ):
                    self.image_list.append(os.path.join(root, fname))
        if len(image_path) == 0:
            msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
            raise RuntimeError(
                msg.format(', '.join(data_extensions), image_path)
            )

        self.preprocess = preprocess

    def __getitem__(self, index):
        img = cv2.imread(self.image_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img

    def __len__(self):
        return len(self.image_list)

