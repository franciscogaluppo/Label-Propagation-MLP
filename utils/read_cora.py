from vertex_features import vertex_features as vf
import numpy as np

data = np.load("../datasets/cora.npz", allow_pickle=True)
adj = csr((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])
attr = csr((data['attr_data'], data['attr_indices'], data['attr_indptr']), shape=data['attr_shape'])
labels = data['labels']
