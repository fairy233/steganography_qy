import cv2
import os
from utils.util import process_path


def imageDeduplication():
    image_path = '/media/a3080/b696e7b7-1f6d-4f3e-967e-d164ff107a68/qiao/HDR_gt'
    image_path = process_path(image_path)
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

    new_list = set(image_list)
    new_list = list(new_list)

    for i in range(len(new_list)):
        cv2.imwrite('/media/a3080/b696e7b7-1f6d-4f3e-967e-d164ff107a68/qiao/HDR512/i.hdr', i)

imageDeduplication()