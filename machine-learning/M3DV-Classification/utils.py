import collections
from itertools import repeat
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import keras.backend as K


class DiceLoss:
    def __init__(self, beta=1., smooth=1.):
        self.__name__ = 'dice_loss_' + str(int(beta * 100))
        self.beta = beta  # the more beta, the more recall
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        bb = self.beta * self.beta
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        weighted_union = bb * K.sum(y_true_f, axis=-1) + \
                         K.sum(y_pred_f, axis=-1)
        score = -((1 + bb) * intersection + self.smooth) / \
                (weighted_union + self.smooth)
        return score


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f' % (epoch+1, score))


def plot_voxel(arr, aux=None):
    if aux is not None:
        assert arr.shape == aux.shape
    length = arr.shape[0]
    _, axes = plt.subplots(length, 1, figsize=(4, 4 * length))
    for i, ax in enumerate(axes):
        ax.set_title("@%s" % i)
        ax.imshow(arr[i], cmap=plt.cm.gray)
        if aux is not None:
            ax.imshow(aux[i], alpha=0.3)
    plt.show()


def plot_voxel_save(path, arr, aux=None):
    if aux is not None:
        assert arr.shape == aux.shape
    length = arr.shape[0]
    for i in range(length):
        plt.clf()
        plt.title("@%s" % i)
        plt.imshow(arr[i], cmap=plt.cm.gray)
        if aux is not None:
            plt.imshow(aux[i], alpha=0.2)
        plt.savefig(path + "%s.png" % i)


def plot_voxel_enhance(arr, arr_mask=None, figsize=10, alpha=0.1):  # zyx
    '''borrow from yuxiang.'''
    plt.figure(figsize=(figsize, figsize))
    rows = cols = int(round(np.sqrt(arr.shape[0])))
    img_height = arr.shape[1]
    img_width = arr.shape[2]
    assert img_width == img_height
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)
    if arr_mask is not None:
        res_mask_img = np.zeros(
            (rows * img_height, cols * img_width), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if (row * cols + col) >= arr.shape[0]:
                continue
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height,
            target_x:target_x + img_width] = arr[row * cols + col]
            if arr_mask is not None:
                res_mask_img[target_y:target_y + img_height,
                target_x:target_x + img_width] = arr_mask[row * cols + col]
    plt.imshow(res_img, plt.cm.gray)
    if arr_mask is not None:
        plt.imshow(res_mask_img, alpha=alpha)
    plt.show()


def find_edges(mask, level=0.5):
    edges = find_contours(mask, level)[0]
    ys = edges[:, 0]
    xs = edges[:, 1]
    return xs, ys


def plot_contours(arr, aux, level=0.5, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(arr, cmap=plt.cm.gray)
    xs, ys = find_edges(aux, level)
    ax.plot(xs, ys)


def crop_at_zyx_with_dhw(voxel, zyx, dhw, fill_with):
    '''Crop and pad on the fly.'''
    shape = voxel.shape
    # z, y, x = zyx
    # d, h, w = dhw
    crop_pos = []
    padding = [[0, 0], [0, 0], [0, 0]]
    for i, (center, length) in enumerate(zip(zyx, dhw)):
        assert length % 2 == 0
        # assert center < shape[i] # it's not necessary for "moved center"
        low = round(center) - length // 2
        high = round(center) + length // 2
        if low < 0:
            padding[i][0] = int(0 - low)
            low = 0
        if high > shape[i]:
            padding[i][1] = int(high - shape[i])
            high = shape[i]
        crop_pos.append([int(low), int(high)])
    cropped = voxel[crop_pos[0][0]:crop_pos[0][1], crop_pos[1]
                                                   [0]:crop_pos[1][1], crop_pos[2][0]:crop_pos[2][1]]
    if np.sum(padding) > 0:
        cropped = np.lib.pad(cropped, padding, 'constant',
                             constant_values=fill_with)
    return cropped


def window_clip(v, window_low=-1024, window_high=400, dtype=np.uint8):
    '''Use lung windown to map CT voxel to grey.'''
    # assert v.min() <= window_low
    return np.round(np.clip((v - window_low) / (window_high - window_low) * 255., 0, 255)).astype(dtype)


def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = scipy.ndimage.interpolation.zoom(voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing


def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reorder(array, order):
    """Reorder the axes"""
    array = array.transpose(order)
    return array


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def mirror_flip(arr, axis, crop_size=32, half_size=16):
    length = arr.shape[0]
    arr_ret = np.ones((length, crop_size, crop_size, crop_size))
    for i in range(0, length):
        tmp_arr = arr[i].copy()
        if axis == 1:
            for j in range(0, half_size):
                tmp_arr[j, :, :], tmp_arr[crop_size - 1 - j, :, :] = tmp_arr[crop_size - 1 - j, :, :], tmp_arr[j, :, :]
            for j in range(0, half_size):
                tmp_arr[:, j, :], tmp_arr[:, crop_size - 1 - j, :] = tmp_arr[:, crop_size - 1 - j, :], tmp_arr[:, j, :]
        elif axis == 2:
            for j in range(0, half_size):
                tmp_arr[:, j, :], tmp_arr[:, crop_size - 1 - j, :] = tmp_arr[:, crop_size - 1 - j, :], tmp_arr[:, j, :]
            for j in range(0, half_size):
                tmp_arr[:, :, j], tmp_arr[:, :, crop_size - 1 - j] = tmp_arr[:, :, crop_size - 1 - j], tmp_arr[:, :, j]
        else:
            for j in range(0, half_size):
                tmp_arr[:, :, j], tmp_arr[:, :, crop_size - 1 - j] = tmp_arr[:, :, crop_size - 1 - j], tmp_arr[:, :, j]
            for j in range(0, half_size):
                tmp_arr[j, :, :], tmp_arr[crop_size - 1 - j, :, :] = tmp_arr[crop_size - 1 - j, :, :], tmp_arr[j, :, :]
        arr_ret[i] = tmp_arr
    return arr_ret


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx


def get_uniform_assign(length, subset):
    assert subset > 0
    per_length, remain = divmod(length, subset)
    total_set = np.random.permutation(list(range(subset)) * per_length)
    remain_set = np.random.permutation(list(range(subset)))[:remain]
    return list(total_set) + list(remain_set)


def split_validation(df, subset, by):
    df = df.copy()
    for sset in df[by].unique():
        length = (df[by] == sset).sum()
        df.loc[df[by] == sset, 'subset'] = get_uniform_assign(length, subset)
    df['subset'] = df['subset'].astype(int)
    return df


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)