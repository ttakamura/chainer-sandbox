import numpy as np
from chainer import Variable, FunctionSet
import chainer.functions as F

class CharIRNN(FunctionSet):
    def __init__(self, n_vocab, n_units):
        super(CharIRNN, self).__init__(
            embed = F.EmbedID(n_vocab, n_units),
            l1_x  = F.Linear(n_units, n_units),
            l1_h  = F.Linear(n_units, n_units),
            l2_h  = F.Linear(n_units, n_units),
            l2_x  = F.Linear(n_units, n_units),
            l3    = F.Linear(n_units, n_vocab),
        )
        for param in self.parameters:
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)
        # IRNN
        self.l1_h.W = np.eye(self.l1_h.W.shape[0], dtype=np.float32) * 0.4
        self.l2_h.W = np.eye(self.l2_h.W.shape[0], dtype=np.float32) * 0.4

    def forward_one_step(self, x_data, y_data, state, train=True, dropout_ratio=0.5):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h0      = self.embed(x)
        h1_in   = self.l1_x(F.dropout(h0, ratio=dropout_ratio, train=train)) + self.l1_h(state['h1'])
        h1      = F.leaky_relu(h1_in)
        h2_in   = self.l2_x(F.dropout(h1, ratio=dropout_ratio, train=train)) + self.l2_h(state['h2'])
        h2      = F.leaky_relu(h2_in)
        y       = self.l3(F.dropout(h2, ratio=dropout_ratio, train=train))
        state   = {'h1': h1, 'h2': h2}

        return state, F.softmax_cross_entropy(y, t)

    def predict(self, x_data, state):
        x = Variable(x_data, volatile=True)

        h0      = self.embed(x)
        h1_in   = self.l1_x(h0) + self.l1_h(state['h1'])
        h1      = F.leaky_relu(h1_in)
        h2_in   = self.l2_x(h1) + self.l2_h(state['h2'])
        h2      = F.leaky_relu(h2_in)
        y       = self.l3(h2)
        state   = {'h1': h1, 'h2': h2}

        return state, F.softmax(y)

    def make_initial_state(self, n_units, batchsize=50, train=True):
        return {name: Variable(np.zeros((batchsize, n_units), dtype=np.float32),
                volatile=not train)
                for name in ('h1', 'h2')}
