import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from CharLSTM import CharLSTM
from CharIRNN import CharIRNN
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def grid_search(model, tuned_parameters, X_train, y_train, X_test, y_test, score='accuracy', n_jobs=-1):
    print '\n' + '='*50
    print score
    print '='*50

    clf = GridSearchCV(model, tuned_parameters, cv=3, scoring=score, n_jobs=n_jobs)
    clf.fit(X_train, y_train)

    print "\n+ Best:\n"
    print clf.best_estimator_

    print "\n+ Avg-score in train-data:\n"
    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)

    print "\n+ Score in test-data:\n"
    y_pred = clf.predict(X_test)
    print classification_report(y_test, y_pred)

    print '\n+ Confussion matrix:\n'
    print confusion_matrix(y_test, y_pred)

    return clf

# --------------------------------------------------------------------------
class BaseChainerEstimator(BaseEstimator):
    def __init__(self, epochs=1000, batch_size=100, report=0, threshold=1e-10, gpu=-1,
                       opt_type='sgd', opt_lr=0.001):
        self.epochs     = epochs
        self.batch_size = batch_size
        self.report     = report
        self.threshold  = threshold
        self.opt_type   = opt_type
        self.opt_lr     = opt_lr
        self.gpu        = gpu
        self.param_names = ['epochs', 'batch_size', 'threshold', 'opt_type', 'opt_lr']

    def _get_param_names(self):
        return self.param_names

    def setup_network(self, n_features):
        error("Not yet implemented")

    def setup_optimizer(self):
        if self.opt_type == 'sgd':
            self.optimizer = optimizers.SGD(lr=self.opt_lr)
        elif self.opt_type == 'adagrad':
            self.optimizer = optimizers.AdaGrad(lr=self.opt_lr)
        elif self.opt_type == 'adam':
            self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.network.collect_parameters())

    def forward(self, x):
        error("Not yet implemented")
        # return self.network.l1(x)

    def loss_func(self, y, t):
        error("Not yet implemented")
        # return F.mean_squared_error(y, t)

    def predict_func(self, y):
        error("Not yet implemented")
        # return F.identity(y)

    def fit(self, x_data, y_data):
        n_samples  = x_data.shape[0]
        n_features = x_data.shape[1]
        score      = 0.0
        self.converge = False
        self.setup_network(n_features)
        if self.gpu >= 0:
            cuda.init()
            self.network.to_gpu()
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)
        self.setup_optimizer()
        for epoch in xrange(self.epochs):
            if not self.converge:
                for i in xrange(n_samples / self.batch_size):
                    x, t  = self.make_batch(x_data, y_data, i)
                    loss  = self.fit_update(x, t)
                score = self.fit_report(epoch, loss, score)
        return self

    def make_batch(self, x_data, y_data, i):
        first = i     * self.batch_size
        last  = (i+1) * self.batch_size
        x     = Variable(x_data[first:last])
        t     = Variable(y_data[first:last])
        return x, t

    def fit_update(self, x, t):
        self.optimizer.zero_grads()
        loss = self.loss_func(self.forward(x), t)
        loss.backward()
        self.optimizer.update()
        return loss

    def fit_report(self, epoch, loss, prev_score):
        score = cuda.to_cpu(loss).data
        score_diff = prev_score - score
        if not score_diff == 0.0 and abs(score_diff) < self.threshold:
            # print("Finish fit iterations!! ==========================")
            self.converge = True
        if self.report > 0 and epoch % self.report == 0:
            self.print_report(epoch, score, score_diff)
        return score

    def predict(self, x_data):
        if self.gpu >= 0:
            x_data = cuda.to_gpu(x_data)
        x = Variable(x_data)
        y = self.forward(x)
        return cuda.to_cpu(self.predict_func(y)).data

    def print_report(self, epoch, loss, score):
        print("epoch: {0}, loss: {1}, diff: {2}".format(epoch, loss[0], score[0]))

class ChainerRegresser(BaseChainerEstimator, RegressorMixin):
    pass

class ChainerClassifier(BaseChainerEstimator, ClassifierMixin):
    def __init__(self, **params):
        BaseChainerEstimator.__init__(self, **params)

    def predict(self, x_data):
        return BaseChainerEstimator.predict(self, x_data).argmax(1)

    def loss_func(self, y, t):
        return F.softmax_cross_entropy(y, t)

    def predict_func(self, h):
        return F.softmax(h)

# --------------------------------------------------------------------------
class LogisticRegression(ChainerClassifier):
    def __init__(self, net_hidden=100, net_out=5, **params):
        ChainerClassifier.__init__(self, **params)
        self.net_hidden = net_hidden
        self.net_out    = net_out
        self.param_names.append('net_hidden')
        self.param_names.append('net_out')

    def setup_network(self, n_features):
        self.network = FunctionSet(
            l1 = F.Linear(n_features,      self.net_hidden),
            l2 = F.Linear(self.net_hidden, self.net_out)
        )

    def forward(self, x):
        h = F.relu(self.network.l1(x))
        y = self.network.l2(h)
        return y
