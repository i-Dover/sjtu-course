import os
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from utils import RocAucEvaluation
from models.densenet import get_compiled
from dataset import ClfDataset, get_balanced_loader
from utils import rotation, reflection, mirror_flip


config = tf.ConfigProto()  # if tf2, can use tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)  # if tf2, can use tf.compat.v1.Session()

# set path
TRAIN_VAL_PATH = 'F:/data/sjtu-ee228-2020/train_val/train_val'
TRAIN_VAL_INFO = 'F:/data/sjtu-ee228-2020/train_val.csv'
TEST_PATH = 'F:/data/sjtu-ee228-2020/test/test'
TEST_INFO = 'F:/data/sjtu-ee228-2020/sampleSubmission.csv'

# init some parameters
NUM_CLASSES = 2
batch_size = 32
epochs = 10
crop_size = 32
half_size = int(crop_size / 2)
resume = True  # Whether load weights
train = True  # Whether to train
train_mode = 1
model_path = 'weights/weights.0.h5'  # the pre-trained weights


def get_train_data():
    """
    Get the train voxel data and crop to [32, 32, 32], and set the data to a numpy.ndarray
    which is faster for training than the dataloader.
    Returns:
        X_train(np.ndarry): [465, 32, 32, 32]
        X_val(np.ndarray): [465, 32, 32, 32]
        y_train(np.ndarray): [465, 2]
        y_val(np.ndarray): [465, 2]
    """
    # read train data
    i = 0
    X_train = np.ones((465, crop_size, crop_size, crop_size))
    train_val = pd.read_csv(TRAIN_VAL_INFO).values
    filenames = train_val[:, 0].tolist()
    for filename in filenames:
        npz = np.load(os.path.join(TRAIN_VAL_PATH, filename + '.npz'))
        voxel = npz['voxel']
        seg = npz['seg']
        X_train[i] = (voxel * seg)[50 - half_size:50 + half_size, 50 - half_size:50 + half_size,
                      50 - half_size:50 + half_size]  # center crop
        i += 1
    # read train label
    y_train = np.loadtxt(TRAIN_VAL_INFO, int, delimiter=",", skiprows=1, usecols=1)
    # set validation
    X_val = X_train.copy()
    y_val = y_train.copy()
    # change the ndarray shape
    return X_train, y_train, X_val, y_val


def get_test_data():
    """
        Get the test voxel data and crop to [32, 32, 32], and set the data to a numpy.ndarray
        which is faster for predicting than the dataloader.
        Returns:
            X_test(np.ndarry): [117, 32, 32, 32]
        """
    # read test data
    i = 0
    X_test = np.ones((117, crop_size, crop_size, crop_size))
    sampleSub = pd.read_csv(TEST_INFO).values
    filenames = sampleSub[:, 0].tolist()
    for filename in filenames:
        npz = np.load(os.path.join(TEST_PATH, filename + '.npz'))
        voxel = npz['voxel']
        seg = npz['seg']
        X_test[i] = (voxel * seg)[50 - half_size:50 + half_size, 50 - half_size:50 + half_size,
                    50 - half_size:50 + half_size]
        i += 1
    return X_test


def expand_dim(X):
    """
    Expand the last dimension.
    Args:
        X(np.ndarray): [dim0, dim1, dim2, dim3]
    Returns:
        np.ndarray: [dim0, dim1, dim2, dim3, 1]
    """
    return X.reshape(X.shape[0], crop_size, crop_size, crop_size, 1)


def augmentation(X_train, y_train):
    """
    Do mirror flip, rotation and reflection data augmentation.
    """
    length = X_train.shape[0]
    tmp_X = X_train.copy()
    tmp_y = y_train.copy()
    X_train = np.ones((length * 3, crop_size, crop_size, crop_size))
    y_train = np.ones(length * 3)

    def transform(arr):
        angle = np.random.randint(4, size=3)
        arr_ret = rotation(arr, angle=angle)
        axis = np.random.randint(4) - 1
        arr_ret = reflection(arr_ret, axis=axis)
        return arr_ret

    X_train[length * 0:length * 1] = mirror_flip(tmp_X, axis=1)
    X_train[length * 1:length * 2] = mirror_flip(tmp_X, axis=2)
    X_train[length * 2:length * 3] = mirror_flip(tmp_X, axis=3)
    for i in range(0, X_train.shape[0]):
        X_train[i, :, :, :] = transform(X_train[i, :, :, :])
    y_train[length * 0:length * 1] = tmp_y.copy()
    y_train[length * 1:length * 2] = tmp_y.copy()
    y_train[length * 2:length * 3] = tmp_y.copy()
    return X_train, y_train


def main():
    # compile model
    model = get_compiled()

    # load weights
    if resume:
        model.load_weights(model_path)
        X_train, y_train, X_val, y_val = get_train_data()
        X_train, y_train = augmentation(X_train, y_train)
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_val = to_categorical(y_val, NUM_CLASSES)
        X_train = expand_dim(X_train)
        X_val = expand_dim(X_val)
        print('Shape of training data: ', X_train.shape, y_train.shape)
        print('Shape of validation data: ', X_val.shape, y_val.shape)
    else:
        X_train, y_train, X_val, y_val = get_train_data()
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_val = to_categorical(y_val, NUM_CLASSES)
        X_train = expand_dim(X_train)
        X_val = expand_dim(X_val)
        print('Shape of training data: ', X_train.shape, y_train.shape)
        print('Shape of validation data: ', X_val.shape, y_val.shape)

    # train
    if train:
        if train_mode == 1:  # do augmentation on-fly
            train_dataset = ClfDataset(crop_size=32, move=3)
            train_loader = get_balanced_loader(train_dataset, batch_sizes=[16, 16])
            roc_auc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
            checkpointer = ModelCheckpoint(filepath='tmp/weights.{epoch:02d}.h5', verbose=1,
                                           period=1, save_weights_only=True)
            model.fit_generator(generator=train_loader, steps_per_epoch=len(train_dataset), max_queue_size=500,
                                workers=1, validation_data=(X_val, y_val), epochs=epochs,
                                callbacks=[checkpointer, roc_auc])

        elif train_mode == 2:  # pre-done augmentation
            roc_auc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
            checkpointer = ModelCheckpoint(filepath='tmp/weights.{epoch:02d}.h5', verbose=1,
                                           period=1, save_weights_only=True)
            model.fit(X_train,
                      y_train,
                      epochs=epochs,
                      validation_data=(X_val, y_val),
                      shuffle=False,
                      batch_size=batch_size,
                      callbacks=[roc_auc, checkpointer])

    # predict
    X_test = get_test_data()
    X_test = expand_dim(X_test)
    print('Shape of test data: ', X_test.shape)
    test_pred = model.predict(X_test, batch_size, verbose=1)

    # save submission
    submission = pd.read_csv(TEST_INFO).values
    submission[:, 1] = test_pred[:, 1]
    df = pd.DataFrame(submission, columns=['name', 'predicted'])
    df.to_csv('Submission.csv', index=None)
    print('File Saved.')


if __name__ == '__main__':
    main()
