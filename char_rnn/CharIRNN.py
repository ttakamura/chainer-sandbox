import six
import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F

class CharIRNN(FunctionSet):
    def __init__(self, n_vocab, n_units, batch_size):
        super(CharIRNN, self).__init__(
            embed = F.EmbedID(n_vocab, n_units),
            l1_x  = F.Linear(n_units, n_units),
            l1_h  = F.Linear(n_units, n_units),
            l2_h  = F.Linear(n_units, n_units),
            l2_x  = F.Linear(n_units, n_units),
            l3    = F.Linear(n_units, n_vocab),
        )
        self.sorted_funcs = sorted(six.iteritems(self.__dict__))
        for param in self.parameters:
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)
        self.l1_h.W = np.eye(self.l1_h.W.shape[0], dtype=np.float32) * 0.5
        self.l2_h.W = np.eye(self.l2_h.W.shape[0], dtype=np.float32) * 0.5
        self.reset_state(batch_size)

    def _get_sorted_funcs(self):
        return self.sorted_funcs

    # def to_gpu
    # TODO
    # if self.gpu >= 0:
    #     for key, value in self.network.state.items():
    #         value.data = cuda.to_gpu(value.data)

    def forward(self, x, state, train=True, dropout_ratio=0.5):
        # x.volatile = not train

        h0 = self.embed(x)
        if dropout_ratio > 0.1:
            h0 = F.dropout(h0, ratio=dropout_ratio, train=train)

        h1 = F.leaky_relu(self.l1_x(h0) + self.l1_h(state['h1']))
        if dropout_ratio > 0.1:
            h1 = F.dropout(h1, ratio=dropout_ratio, train=train)

        h2 = F.leaky_relu(self.l2_x(h1) + self.l2_h(state['h2']))
        if dropout_ratio > 0.1:
            h2 = F.dropout(h2, ratio=dropout_ratio, train=train)

        y = self.l3(h2)
        return {'h1': h1, 'h2': h2}, y

    def train(self, x, t, dropout_ratio=0.5):
        new_state, y = self.forward(x, self.state, train=True, dropout_ratio=0.0)
        self.state = new_state
        return F.softmax_cross_entropy(y, t)

    def predict(self, x):
        new_state, y = self.forward(x, self.state, train=False, dropout_ratio=0.0)
        self.state = new_state
        return F.softmax(y)

    def make_initial_state(self, n_units, batch_size=50, train=True):
        return {name: Variable(np.zeros((batch_size, n_units), dtype=np.float32), volatile=not train)
                for name in ('h1', 'h2')}

    def reset_state(self, batch_size):
        self.state = self.make_initial_state(self.l1_h.W.shape[0], batch_size=batch_size)
