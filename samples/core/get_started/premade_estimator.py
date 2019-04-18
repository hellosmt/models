#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser() #a rgparse 是Python内建的用于解析命令行参数的模块
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:]) # argv=['premade_estimator.py', '--batch_size', '32', '--train_steps', '1000']
    # args = Namespace(batch_size=32, train_steps=1000)， 后面获得batch_size可以用args.batch_size

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    print(iris_data.load_data())

    # Feature columns describe how to use the input.
    my_feature_columns = []
    # 这些键值也是特征点的名称。把特征点的名称告诉模型，模型才能使用。
    # 特征列是一种数据结构，告知模型如何解读每个特征中的数据，不仅要告诉模型特征的名称，还要告诉模型特征的数据类型。
    # 上面打印出来可以看到train_x每列的键值是['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']
    # 若特征列中的特征名字与train_x中的特征名字不一致，则模型无法找到特征数据。
    for key in train_x.keys(): 
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        # 为什么要用到lambda？因为input_fn必须是一个函数对象，而不是函数调用的返回值：input_fn=my_input_fn()，
        # 如果带参数则会报类型错误的错，所以要用lambda进行封装
        # train_x是DataFrame类型的变量 train_y是Series类型的变量
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size), 
        steps=args.train_steps)

    # Evaluate the model.
    # 由于evaluate只运行一次，而且需要优化神经网络参数，所以，输入给evaluate方法的数据，不需要随机化（shuffle）和重复（repeat）操作
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    # 预测是对无标签样本进行预测，所以不用传label进去
    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    # predict 方法返回一个 Python 可迭代对象predictions，为每个样本生成一个预测结果字典
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
