import os
import re
import random
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.utils.data as data
from collections import OrderedDict

from PIL import Image
import pprint

RE_CELL_IMAGE = re.compile(r'^(?P<id>[0-9]+)-.*\.(?P<ext>(bmp|png))$', re.IGNORECASE)

def _mkrange(x):
    if isinstance(x, float):
        return (x, x)
    else:
        return x

class CellsTransform:
    def __init__(self,
            degrees,
            translate,
            scale,
            crop_size,
            brightness=.5,
            contrast=.5,
            saturation=.5,
            hue=.5):
        if isinstance(scale, float):
            scale = (scale, scale)
        self.common_affine = dict(
                degrees=_mkrange(degrees),
                translate=translate,
                scale_ranges=_mkrange(scale),
                shears=None)
        self.common_crop = dict(output_size=_mkrange(crop_size))
        self.color_params = dict(
                brightness=_mkrange(brightness),
                contrast=_mkrange(contrast),
                saturation=_mkrange(saturation),
                hue=_mkrange(hue),)
        self.to_tensor = transforms.ToTensor()
    def __call__(self, x, y):
        affine_params = transforms.RandomAffine.get_params(
                img_size=x.size, # PIL.Image.size
                **self.common_affine)
        x, y = TF.affine(x, *affine_params), TF.affine(y, *affine_params)
        crop_params = transforms.RandomCrop.get_params(x, **self.common_crop)
        # TODO: possibly some custom ops different for x and y
        x, y = TF.crop(x, *crop_params), TF.crop(y, *crop_params)
        flip = random.random() < 0.5
        if flip:
            x, y = TF.vflip(x), TF.vflip(y)
        # TODO: PIL only works with grayscaly images...
        # color_transform = transforms.ColorJitter.get_params(**self.color_params)
        # x, y = color_transform(x), color_transform(y)
        # RETURNS PIL Image
        return x, y
    def __repr__(self):
        return pprint.pformat(dict(
            common_affine=self.common_affine,
            common_crop=self.common_crop,
            vflip_prob=0.5,
            ))

class CellsSegmentation(data.Dataset):
    def __init__(
            self,
            subset,
            clone_times,
            source_path='BBBC018_v1_images-fixed',
            target_path='BBBC018_v1_outlines',
            xy_transform=None
            ):
        assert subset in ['train', 'test', 'val']
        self.clone_times = clone_times
        source_path, target_path = [os.path.join(bd, subset) for bd in (source_path, target_path)]
        self.source_path = source_path
        self.target_path = target_path
        self.xy_transform = xy_transform
        self.i2x = list() # maps serial id into source image
        self.i2y = list() # maps serial id into ground truth segmentation
        self.uuid2i = dict() # maps identifying prefix of filename into serial id
        self.i2uuid = list()
        for i2path, basedir in [
                (self.i2x, source_path),
                (self.i2y, target_path),
                ]:
            for dirpath, _, fnames in os.walk(basedir):
                dirpath = os.path.abspath(dirpath)
                for fname in fnames:
                    filename = os.path.join(dirpath, fname)
                    match = RE_CELL_IMAGE.match(fname)
                    if not match:
                        continue
                    uuid = match.group('id')
                    i = self.uuid2i.get(uuid, len(i2path))
                    self.uuid2i[uuid] = i
                    while i >= len(i2path):
                        i2path.append(None)
                        self.i2uuid.append(None)
                    self.i2uuid[i] = uuid
                    i2path[i] = os.path.join(dirpath, filename)

    def __getitem__(self, index):
        index = index % len(self.i2x)
        x = self.i2x[index]
        y = self.i2y[index]
        x = Image.open(x).convert('RGB')
        y = Image.open(y)
        if self.xy_transform is not None:
            x, y = self.xy_transform(x, y)
        x, y = TF.to_tensor(x), TF.to_tensor(y)
        y = (y > 0).byte()
        return x, y

    def __repr__(self):
        return pprint.pformat(OrderedDict(
            name=self.__class__.__name__,
            source_path=self.source_path,
            target_path=self.target_path,
            n_samples=len(self.i2x),
            n_samples_augmented=len(self),
            x_transform=self.x_transform.__class__.__name__,
            y_transform=self.y_transform.__class__.__name__,
            ))

    def __len__(self):
        return self.clone_times * len(self.i2x)
