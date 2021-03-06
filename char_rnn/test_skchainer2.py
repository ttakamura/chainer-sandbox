import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from chainer import cuda

import skchainer as skc

# cuda.init()

digits = datasets.load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuned_parameters = [
    {'opt_type': ['sgd'],      'opt_lr': [0.01, 0.001], 'net_hidden': [50, 100, 200, 300]},
    {'opt_type': ['adagrad'],  'opt_lr': [0.01, 0.001], 'net_hidden': [50, 100, 200, 300]},
    {'opt_type': ['adam'],                              'net_hidden': [50, 100, 200, 300]}
]
model = skc.LogisticRegressionEstimator(epochs=50, net_out=10, threshold=1e-6)

skc.grid_search(model, tuned_parameters, X_train, y_train, X_test, y_test, score='accuracy')
