import cv2
import os
from utils.util import process_path


def imageDeduplication():
    image_path = 'hdr/test'
    image_path = process_path(image_path)
    data_extensions = ['.hdr', '.exr', '.jpeg']
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
        cv2.imwrite('%s/hdr512/i.hdr' % ( image_path ), i)

imageDeduplication()