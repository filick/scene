import torchvision.datasets as td
from PIL import Image
import os
import json


__all__ = ['TRAIN_ROOT', 'VALIDATION_ROOT', 'TEST_ROOT', 'ChallengerSceneFolder']


TRAIN_ROOT = None
VALIDATION_ROOT = None
TEST_ROOT = None

current_root = os.path.split(os.path.realpath(__file__))[0]
for item in os.listdir(current_root):
    path = os.path.join(current_root, item)
    if os.path.isdir(path):
        if 'test' in item:
            TEST_ROOT = path
        elif 'validation' in item:
            VALIDATION_ROOT = path
        elif 'train' in item:
            TRAIN_ROOT = path


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ChallengerSceneFolder(td.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        json_label = None
        image_folder = None
        imgs = []
        root = os.path.expanduser(root)
        for item in os.listdir(root):
            path = os.path.join(root, item)
            if os.path.isdir(path):
                image_folder = path
            elif item.endswith('.json'):
                json_label = path

        if json_label is not None:
            with open(json_label, 'r') as f:
                label_list = json.load(f)
            for image in label_list:
                path = os.path.join(image_folder, image['image_id'])
                target = int(image['label_id'])
                imgs.append((path, target))
        else:
            for item in os.listdir(image_folder):
                path = os.path.join(image_folder, item)
                if is_image_file(path):
                    imgs.append((path, -1))

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader