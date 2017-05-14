# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:20:43 2017

@author: ASY

tensorflow guide quickStart

官网上的Iris数据集分类的例子
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

#if 'session' in locals() and session is not None:
#    print('Close interactive session')
#    session.close()
# Data sets
IRIS_TRAINING = r"D:\DL\Tensorflow\Data\iris_training.csv"
IRIS_TEST = r"D:\DL\Tensorflow\Data\iris_test.csv"

# Load dataset
'''
这里默认把最后一列作为label，前面的数据作为feature
'''
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
filename = IRIS_TRAINING,
target_dtype = np.int,
features_dtype = np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
filename = IRIS_TEST,
target_dtype=np.int,
features_dtype = np.float32)

'''
确认所有的特征都有实值
'''
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = 4)]
classifier = tf.contrib.learn.DNNClassifier(
                                            feature_columns = feature_columns,
                                            hidden_units = [10, 20, 10],
                                            n_classes = 3,
                                            model_dir = r"D:\DL\Tensorflow\Data\tmp\iris_model")

classifier.fit(x = training_set.data, y = training_set.target, steps = 20)

# Fit model
classifier.fit(x = training_set.data, y = training_set.target, steps=20)
#classifier.fit(x = training_set.data, y = training_set.target, steps=1000)

accuracy_score = classifier.evaluate(x = test_set.data, y = test_set.target)["accuracy"]
print("Accuracy: {0:f}".format(accuracy_score))


# Classify two new flower samples
new_samples = np.array(
[[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]],dtype = float)
y = list(classifier.predict(new_samples, as_iterable = True))
print('Predictions:{}'.format(str(y)))

