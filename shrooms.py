#!/usr/bin/env python

from __future__ import print_function

try:
    import matplotlib

    matplotlib.use('Agg')
except ImportError:
    pass

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import datasets
from chainer import training
from chainer.training import extensions

import numpy as np
import sklearn.preprocessing as sp

data_array = np.genfromtxt(
    'mushrooms.csv', delimiter=',', dtype=str, skip_header=1)
labelEncoder = sp.LabelEncoder()
for col in range(data_array.shape[1]):
    data_array[:, col] = labelEncoder.fit_transform(data_array[:, col])

X = data_array[:, 1:].astype(np.float32)
Y = data_array[:, 0].astype(np.int32)[:, None]
train, test = datasets.split_dataset_random(
    datasets.TupleDataset(X, Y), int(data_array.shape[0] * .7))

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(
    test, 100, repeat=False, shuffle=False)

class LinearBlock(chainer.Chain):

    def __init__(self):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__(
            fc=L.Linear(n_units))
        )

# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the input size to each layer inferred from the layer before
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l21 = L.Linear(n_units)  # n_units -> n_units
            self.l22 = L.Linear(n_units)  # n_units -> n_units
            self.l23 = L.Linear(n_units)  # n_units -> n_units
            self.l24 = L.Linear(n_units)  # n_units -> n_units
            self.l25 = L.Linear(n_units)  # n_units -> n_units
            self.l26 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h21 = F.relu(self.l2(h2))
        h22 = F.relu(self.l2(h21))
        h23 = F.relu(self.l2(h22))
        h24 = F.relu(self.l2(h23))
        h25 = F.relu(self.l2(h24))
        h26 = F.relu(self.l2(h25))
        return self.l3(h26)


model = L.Classifier(
    MLP(44, 1), lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)

# Setup an optimizer
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

gpu_id = -1  # Change to -1 to use CPU

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
trainer = training.Trainer(updater, (50, 'epoch'), out='result')

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Save two plot images to the result dir
if extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))

# Print selected entries of the log to stdout
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Run the training
trainer.run()

for _ in range(10):
    x, t = test[np.random.randint(len(test))]
    predict = model.predictor(x[None]).data
    predict = predict[0][0] >= 0

    print('Predicted', 'Edible' if predict == 0 else 'Poisonous',
          'Actual', 'Edible' if t[0] == 0 else 'Poisonous')
