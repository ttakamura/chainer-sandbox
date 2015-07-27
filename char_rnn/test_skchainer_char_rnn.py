import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from chainer import cuda
import skchainer as skc

document = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 1000)
# cuda.init()

def parse(words, vocab):
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

doc_len = len(document)
vocab   = {}
dataset = parse(document, vocab)

X = dataset[0:doc_len-1].reshape(doc_len-1, 1)
y = dataset[1:doc_len]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuned_parameters = [
    {'net_type': ['irnn'], 'opt_type': ['adam'], 'net_hidden': [50, 100, 200, 300]}
]
model = skc.RNNCharEstimator(epochs=2, batch_size=10, vocab_size=len(vocab), threshold=1e-6)

skc.grid_search(model, tuned_parameters, X_train, y_train, X_test, y_test, score='accuracy', n_jobs=8)

# predict -----------------------
print model.predict(X[1:5,])
