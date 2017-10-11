# mini PCA

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

data = [
    Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),
    Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),
    Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0])
]
rows = sc.parallelize(data)

mat = RowMatrix(rows) # mat is 3 x 5
pc = mat.computePrincipalComponents(2) # pc is 5 x 2
pc
projected = mat.multiply(pc) # projected is 3 x 2
projected.rows.collect()

# center
means = rows.reduce(lambda a, b: a + b) / rows.count()
centeredRows = rows.map(lambda r: r - means)
centeredMat = RowMatrix(centeredRows)

# then run SVD
svd = centeredMat.computeSVD(2, True)
svd.V # is the same as PC

s = svd.U.rows.sample(False, 0.5)

x = svd.U.multiply(Matrices.dense(2, 2, (svd.s[0], 0, 0, svd.s[1])))
x.rows.collect()