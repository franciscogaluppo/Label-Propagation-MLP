import numpy as np
from Label-Propagation import learn

n_unlabeled, n_labeled, n_train = 500, 100, 100

X_labeled, y_labeled = generate_data2(n_labeled)
X_unlabeled, y_unlabeled = generate_data2(n_unlabeled)
X_train, y_train = generate_data2(n_train)
