import os
import re
import torch.data.utils.data as data
from collections import OrderedDict
import pprint

RE_CELL_IMAGE = re.compile(r'^(?P<id>[0-9]+)-.*\.(?P<ext>(bmp|png))$', re.IGNORECASE)

class CellsSegmentation(data.Dataset):
    def __init__(
            self,
            subset,
            source_path='BBBC018_v1_images-fixed',
            target_path='BBBC018_v1_outlines',
            transform=None,
            target_transform=None):
        assert subset in ['train', 'test', 'val']
        source_path, target_path = [os.path.join(bd, subset) for bd in (source_path, target_path)]
        self.source_path = source_path
        self.target_path = target_path
        self.x_transform = transform
        self.y_transform = target_transform
        self.i2x = list() # maps serial id into source image
        self.i2y = list() # maps serial id into ground truth segmentation
        self.uuid2i = dict() # maps identifying prefix of filename into serial id
        for i2path, basedir in [
                (self.i2x, source_path),
                (self.i2y, target_path),
                ]:
            for filename in os.walk(basedir):
                match = RE_CELL_IMAGE.match(filename)
                if not match:
                    continue
                uuid = match.group('id')
                i = self.uuid2i(uuid, default=len(i2path))
                self.uuid2i[uuid] = i
                while serial >= len(serial2path):
                    i2path.append(None)
                    i2uuid.append(None)
                i2uuid[i] = uuid
                i2path[i] = os.path.join(in_path, filename)

    def __getitem__(self, index):
        x = self.i2x[index]
        y = self.i2y[index]
        x = Image.open(x).convert('RGB')
        y = Image.open(y)
        if self.transform:
            x = self.x_transform(x)
            y = self.y_transform(y)
        return x, y

    def __repr__(self):
        return pprint.pformat(OrderedDict(
            name=self.__class__.__name__,
            source_path=self.source_path,
            target_path=self.target_path,
            n_samples=len(self),
            x_transform=self.x_transform.__class__.__name__,
            y_transform=self.y_transform.__class__.__name__,
            ))

    def __len__(self):
        return len(self.i2x)
