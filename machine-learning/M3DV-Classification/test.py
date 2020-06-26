import os
import numpy as np
import pandas as pd
import tensorflow as tf
from models.densenet import get_model
from dataset import ClfTestDataset


config = tf.ConfigProto()  # if tf2, can use tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)  # if tf2, can use tf.compat.v1.Session()

# set path
TEST_PATH = 'F:/data/sjtu-ee228-2020/test/test'  # your test data dir
TEST_INFO = 'F:/data/sjtu-ee228-2020/sampleSubmission.csv'  # your test data info csv file
MODEL_PATH1 = 'weights/weights.1.h5'
MODEL_PATH2 = 'weights/weights.2.h5'

# set some parameters
batch_size = 32
crop_size = 32
half_size = int(crop_size / 2)


def inference():
    # save the test data to the np.ndarray, which is more faster than the dataloader
    X_test = np.ones((117, crop_size, crop_size, crop_size))
    # read test dat
    i = 0
    sampleSub = pd.read_csv(TEST_INFO).values
    filenames = sampleSub[:, 0].tolist()
    for filename in filenames:
        npz = np.load(os.path.join(TEST_PATH, filename + '.npz'))
        voxel = npz['voxel']
        seg = npz['seg']
        X_test[i] = (voxel * seg)[50 - half_size:50 + half_size, 50 - half_size:50 + half_size, 50 - half_size:50 + half_size]
        i += 1
    X_test = X_test.reshape(X_test.shape[0], crop_size, crop_size, crop_size, 1)

    # build model
    model = get_model()

    model.load_weights(MODEL_PATH1)
    test_pred1 = model.predict(X_test, batch_size, verbose=1)
    model.load_weights(MODEL_PATH2)
    test_pred2 = model.predict(X_test, batch_size, verbose=1)
    # weights.8.h5 can get 0.73691

    # save
    sampleSub[:, 1] = (test_pred1[:, 1] + test_pred2[:, 1]) / 2
    df = pd.DataFrame(sampleSub, columns=['name', 'predicted'])
    df.to_csv('Submission.csv', index=None)


# def inference():
#     test_dataset = ClfTestDataset(crop_size=crop_size, move=None)
#     print("Test Dataset: {}".format(len(test_dataset)))
#
#     model = get_model()
#     model.load_weights(MODEL_PATH)
#
#     sampleSub = pd.read_csv(TEST_INFO)
#     for i in range(len(test_dataset)):
#         inputs = test_dataset[i]
#         inputs = np.expand_dims(inputs, axis=0)
#
#         predictions = model.predict(inputs, steps=1)
#         score = predictions[0][1]
#         print(score)  # the score is the prob of label 1
#         sampleSub.loc[i, 'predicted'] = score
#     sampleSub.to_csv('Submission.csv', index=None)


if __name__ == '__main__':
    inference()