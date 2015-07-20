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

    def fit(self, x_data, y_data):
        self.n_samples  = x_data.shape[0]
        self.n_features = x_data.shape[1]
        self.converge   = False

        self.setup_network(self.n_features)
        if self.gpu >= 0:
            cuda.init()
            self.network.to_gpu()
        self.setup_optimizer()

        score = 0.0
        for epoch in xrange(self.epochs):
            if not self.converge:
                for i in xrange(self.n_samples / self.batch_size):
                    x_batch, y_batch = self.make_batch(x_data, y_data, i)
                    if self.gpu >= 0:
                        x = Variable(cuda.to_gpu(x))
                        t = Variable(cuda.to_gpu(t))
                    else:
                        x = Variable(x_batch)
                        t = Variable(y_batch)
                    loss = self.forward_train(x, t)
                    self.fit_update(loss, i)
                score = self.fit_report(epoch, loss, score)
        return self

    def make_batch(self, x_data, y_data, batch_id):
        first   = batch_id     * self.batch_size
        last    = (batch_id+1) * self.batch_size
        x_batch = x_data[first:last]
        y_batch = y_data[first:last]
        return x_batch, y_batch

    def fit_update(self, loss, batch_id):
        self.optimizer.zero_grads()
        loss.backward()
        # self.optimizer.clip_grads(grad_clip)
        self.optimizer.update()

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
        y = self.forward_predict(x)
        return cuda.to_cpu(y).data

    def print_report(self, epoch, loss, score):
        print("epoch: {0}, loss: {1}, diff: {2}".format(epoch, loss[0], score[0]))

class ChainerRegresser(BaseChainerEstimator, RegressorMixin):
    pass

class ChainerClassifier(BaseChainerEstimator, ClassifierMixin):
    def __init__(self, **params):
        BaseChainerEstimator.__init__(self, **params)

    def predict(self, x_data):
        return BaseChainerEstimator.predict(self, x_data).argmax(1)

    def forward_train(self, x, t):
        y = self.forward_inner(x, train=True)
        return F.softmax_cross_entropy(y, t)

    def forward_predict(self, x):
        y = self.forward_inner(x, train=False)
        return F.softmax(y)

# --------------------------------------------------------------------------
class LogisticRegressionEstimator(ChainerClassifier):
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

    def forward_inner(self, x, train=True):
        h = F.relu(self.network.l1(x))
        y = self.network.l2(h)
        return y

# --------------------------------------------------------------------------
class RNNCharEstimator(ChainerClassifier):
    def __init__(self, net_type='lstm', net_hidden=100,
                       vocab_size=1000, dropout_ratio=0.0, seq_size=70, grad_clip=100.0,
                       **params):
        ChainerClassifier.__init__(self, **params)
        self.net_hidden    = net_hidden
        self.net_type      = net_type
        self.vocab_size    = vocab_size
        self.dropout_ratio = dropout_ratio
        self.seq_size      = seq_size
        self.grad_clip     = grad_clip
        self.param_names.append('net_type')
        self.param_names.append('net_hidden')
        self.param_names.append('dropout_ratio')

    def setup_network(self, n_features):
        if self.net_type == 'lstm':
            self.network = CharLSTM(self.vocab_size, self.net_hidden)
        elif self.net_type == 'irnn':
            self.network = CharIRNN(self.vocab_size, self.net_hidden)
        else:
            error("Unknown net_type")

        self.state = self.network.make_initial_state(self.net_hidden, batch_size=self.batch_size)
        if self.gpu >= 0:
            for key, value in self.state.items():
                value.data = cuda.to_gpu(value.data)

        self.reset_accum_loss()

    def reset_accum_loss(self):
        if self.gpu >= 0:
            self.accum_loss = Variable(cuda.zeros(()))
        else:
            self.accum_loss = Variable(np.zeros(()))

    def forward_train(self, x, t):
        new_state, loss = self.network.train(x, t, self.state, dropout_ratio=self.dropout_ratio)
        self.state = new_state
        return loss

    def forward_predict(self, x):
        new_state, prediction = self.network.predict(x, self.state)
        self.state = new_state
        return prediction

    def fit_update(self, loss, batch_id):
        self.accum_loss += loss

        if (batch_id + 1) % self.seq_size == 0: # Run Truncated BPTT
            self.optimizer.zero_grads()
            self.accum_loss.backward()
            self.accum_loss.unchain_backward()  # truncate
            self.optimizer.clip_grads(self.grad_clip)
            self.optimizer.update()
            self.reset_accum_loss()

    def make_batch(self, x_data, y_data, batch_id):
        batch_num = self.n_samples / self.batch_size
        x_batch = np.array([x_data[(batch_id + batch_num * j) % self.n_samples]
                            for j in xrange(self.batch_size)])
        y_batch = np.array([y_data[(batch_id + batch_num * j) % self.n_samples]
                            for j in xrange(self.batch_num)])
        return x_batch, y_batch
