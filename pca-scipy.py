# https://blog.dominodatalab.com/pca-on-very-large-neuroimaging-datasets-using-pyspark/

import numpy as np
from sklearn.decomposition import PCA

x = np.array([[0.0, 1.0, 0.0, 7.0, 0.0], [2.0, 0.0, 3.0, 4.0, 5.0], [4.0, 0.0, 0.0, 6.0, 7.0]])
x.shape # x is 3 x 5

pca = PCA(n_components=2, whiten = False)
pca.fit(x)

# Components (matrix V)
v = pca.components_.transpose()
v
v.shape # v is 5 x 2

# Representation in 2-D (matrix Z)
z = pca.transform(x)
z
z.shape # z is 3 x 2

# Covariance matrix
cxx = pca.get_covariance()
cxx
cxx.shape # cxx is 5 x 5

