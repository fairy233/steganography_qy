import cv2
import os

def in_list(image, new_list):
    for image_temp in new_list:
        if image_temp.shape == image.shape:
            if(image_temp - image).any() == False:
                return False
    return True

def imageDeduplication():
    image_path = '/media/a3080/b696e7b7-1f6d-4f3e-967e-d164ff107a68/qiao/HDR_gt'
    data_extensions = ['.hdr', '.exr']
    name_list = []
    image_list = []

    for root, _, fnames in sorted(os.walk(image_path)):
        for fname in fnames:
            if any(
                    fname.lower().endswith(extension)
                    for extension in data_extensions
            ):
                name_list.append(os.path.join(root, fname))

    for i in range(len(name_list)):
        img = cv2.imread(name_list[i], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        image_list.append(img)
        print("正在载入第" + str(i+1) + "张图片")

    # new_list = set(image_list)
    # new_list = list(new_list)
    new_list = []
    x=0
    y=0
    for i in image_list:
        if in_list(i, new_list):
            new_list.append(i)
            x=x+1
        else:
            y=y+1
        print("正在处理，不重复个数 "+str(x), " 重复个数"+str(y))
        


    j = 0
    for i in new_list:
        cv2.imwrite('/media/a3080/b696e7b7-1f6d-4f3e-967e-d164ff107a68/qiao/HDR512/%s.hdr' % (str(j)), i)
        j = j  + 1
        print("正在保存第" + str(j) + "张图片")

imageDeduplication()
