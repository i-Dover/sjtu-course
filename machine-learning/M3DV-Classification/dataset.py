from collections.abc import Sequence
import random
import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from utils import rotation, reflection, crop, random_center, reorder, resize, _triple, crop_at_zyx_with_dhw

TRAIN_INFO = 'F:/data/sjtu-ee228-2020/train_val.csv'
TRAIN_NODULE_PATH = 'F:/data/sjtu-ee228-2020/train_val/train_val'
TEST_INFO = 'F:/data/sjtu-ee228-2020/sampleSubmission.csv'
TEST_NODULE_PATH = 'F:/data/sjtu-ee228-2020/test/test'
VAL_INFO = 'F:/data/sjtu-ee228-2020/val.csv'

LABEL = [0, 1]


class ClfDataset(Sequence):
    def __init__(self, crop_size=32, move=3):
        """The classification-only dataset.
        Args:
            crop_size(int): the input size
            move(int | None): the random move
        """
        self.transform = Transform(crop_size, move)
        self.df = pd.read_csv(TRAIN_INFO)
        self.label = self.df.values[:, 1]

    def __getitem__(self, idx):
        name = self.df.loc[idx, 'name']
        with np.load(os.path.join(TRAIN_NODULE_PATH, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel']*npz['seg'])
        label = self.label[idx]
        return voxel, label

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        ys = to_categorical(ys, 2)
        xs, ys = mixup(xs, ys, 0.3)  # mixup augmentation
        return xs, ys


class ClfTestDataset(Sequence):
    def __init__(self, crop_size=32, move=None):
        self.transform = Transform(crop_size, move)
        self.df = pd.read_csv(TEST_INFO)

    def __getitem__(self, idx):
        name = self.df.loc[idx, 'name']
        with np.load(os.path.join(TEST_NODULE_PATH, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel']*npz['seg'])
        return voxel

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _collate_fn(data):
        xs = []
        for x in data:
            xs.append(x)
        return np.array(xs)


class ClfValDataset(object):
    def __init__(self, crop_size=32, move=None):
        """The classification-only dataset.
        Args:
            crop_size(int): the input size
            move(int | None): the random move
        """
        self.transform = Transform(crop_size, move)
        self.df = pd.read_csv(VAL_INFO)
        self.label = self.df.values[:, 1]

    def __getitem__(self, idx):
        name = self.df.loc[idx, 'name']
        with np.load(os.path.join(TRAIN_NODULE_PATH, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'] * npz['seg'])
            # voxel = self.transform(npz['voxel'] * (npz['seg'] * 0.8 + 0.2))
        label = self.label[idx]
        return voxel, label

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        ys = to_categorical(ys, 2)
        return xs, ys


def mixup(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def get_balanced_loader(dataset, batch_sizes):
    assert len(batch_sizes) == len(LABEL)
    total_size = len(dataset)
    print('Size', total_size)
    index_generators = []
    for l_idx in range(len(batch_sizes)):
        # this must be list, or `l_idx` will not be eval
        iterator = [i for i in range(total_size) if dataset.label[i] == l_idx]
        index_generators.append(shuffle_iterator(iterator))
    while True:
        data = []
        for i, batch_size in enumerate(batch_sizes):
            generator = index_generators[i]
            for _ in range(batch_size):
                idx = next(generator)
                data.append(dataset[idx])
        yield dataset._collate_fn(data)


class Transform:
    """The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    """

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            # order = np.random.permutation(3)  # generate order
            # arr_ret = reorder(arr_ret, order)
            # zoom_rate = 0.8 + (1.15 - 0.8) * np.random.rand()
            # (arr_ret, _) = resize(arr_ret, [1., 1., 1.], new_spacing=[zoom_rate] * 3)
            # zoomed_center = arr_ret.shape // 2
            # arr_ret = crop_at_zyx_with_dhw(arr_ret, zoomed_center, self.size, 0)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                # aux_ret = reorder(aux_ret, order=order)
                # (aux_ret, _) = resize(aux_ret, [1., 1., 1.], new_spacing=[zoom_rate] * 3)
                # aux_ret = crop_at_zyx_with_dhw(aux_ret, zoomed_center, self.size, 0)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)


if __name__ == '__main__':
    train_dataset = ClfDataset(crop_size=32, move=3)
    train_loader = get_loader(train_dataset, batch_size=8)
    for i in train_loader:
        print(i[1])
