# Medical 3D Voxel Classification
This is the course work of EE228 (machine learning(AI)) of SJTU, Medical 3D Voxel Classification(M3DV) Kaggle inclass competition.  
This is a classification project for pulmonary nodules. We need to train and evaluate our model on a train_val dataset, and then predict results on the test dataset. The final result in Kaggle is 0.73870.

# Code Structure
* [`train.py`](train.py): This is for compiling the model and training it on the train_val dataset.
* [`test.py`](test.py): This is for loading trained weights and predicting results, the submission results will be saved to "Submission.csv".
* [`models/`](mylib/)
    * [`densenet.py`](mylib/densenet.py): 3D *DenseNet* models. Reference from:https://github.com/duducheng/DenseSharp
* [`dataset.py`](dataset.py): Dataset and DataLoader, and there contains Transform class to do data augmentation on the fly.
* [`utils.py`](utils.py): This contains many useful functions, such as roc_auc scorer, data augmentation functions (rotation, reflection, crop, and mirror flip, etc.).
# Dataset Files
* [`datasets/`](datasets/):
    * [`train_val/`](datasets/train_val/): Training dataset.
    * [`test/`](datasets/test/): Test dataset.
    * [`train_val.csv`](datasets/train_val.csv): Labels of training dataset.
    * ***Note**: The dataset is not available publicly, so I don't get all the training and test data, just two samples are provided for demoing ,one in train_val and another in test.*
* [`sampleSubmission.csv`](sampleSubmission.csv): A sample submission file.
# Weight Files

* [`weights/`](weights/): Containing some trained weight files.

# Requirements
* Python 3 (Anaconda 3 specifically)
* TensorFlow==1.15.0
* Keras==2.2.0
