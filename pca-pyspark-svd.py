# https://blog.dominodatalab.com/pca-on-very-large-neuroimaging-datasets-using-pyspark/

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("pca") \
    .getOrCreate()
sc = spark.sparkContext

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

data = [
    Vectors.dense([0.0, 1.0, 0.0, 7.0, 0.0]),
    Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),
    Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),
]
data
rows = sc.parallelize(data)

mat = RowMatrix(rows)
mat.numRows(), mat.numCols() # mat is 3 x 5

# center
means = rows.reduce(lambda a, b: a + b) / rows.count()
centeredRows = rows.map(lambda r: r - means)
centeredMat = RowMatrix(centeredRows)
centeredMat.numRows(), centeredMat.numCols() # centeredMat is 3 x 5

svd = centeredMat.computeSVD(2, True)

# Components (matrix V)
v = svd.V
v # v is 5 x 2

# Representation in 2-D (matrix Z)
z = centeredMat.multiply(v).rows.collect()
z # z is 3 x 2
