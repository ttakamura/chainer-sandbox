import numpy as np
from scipy import special
import skchainer as skc

X = np.linspace(-10, 10, 500).astype(np.float32)
p = special.expit(X)
y = np.random.binomial(1, p).astype(np.int32)
X = X.reshape(len(X), 1)

model = skc.LogisticRegression(net_out=2, threshold=1e-6)
model.fit(X, y)
print(model.score(X, y))
