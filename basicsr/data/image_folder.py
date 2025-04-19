###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns

def read_fns_clean(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().replace(",", "").replace("(", "").replace(")", "").replace("\n", " ").replace(".", "").replace(":", "")
        words = text.split(" ")  # 将文本拆分成单词列表
        fns = ' '.join(words[:50])
    return fns

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, fns=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):                
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')

if __name__ == '__main__':
    t = read_fns_clean("/home/chenxiao/eccv24_reflection_removal/datasets/reflection-dataset/test/SIR2/WildSceneDataset/reflection_captions/034.txt")
    print(t)