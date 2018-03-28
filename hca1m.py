# Start a Spark shell (`pyspark2 --driver-memory 4G --num-executors 16 --executor-cores 1 --executor-memory 4G --driver-memory 8G` on CDH)
# then run the following

from pyspark.sql import SparkSession
  
spark = SparkSession\
  .builder\
  .appName("hca")\
  .getOrCreate()
sc = spark.sparkContext

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

df = spark.read.parquet("celldb1m")
numFeatures = 27998
rows = df.rdd.map(lambda (id, indices, values): (id, Vectors.sparse(numFeatures, indices, values )))
rows.cache()

# Now let's do some simple queries

# 0. Find out how many rows there are
c = rows.count()
print(c)

# 1. Find the number of measurements per sample
numMeasurementsPerSample = rows.mapValues(lambda vec : len(vec.values))
numMeasurementsPerSample.take(5)

# 2. Calculate the sparsity of the whole dataset (7%)
meas = numMeasurementsPerSample.values().mean() / numFeatures
print(meas)

# 3. Find the number of true zeros (not NA) per sample (0)
trueZerosPerSample = rows.mapValues(lambda vec : sum(x == 0.0 for x in vec.values))
trueZerosPerSample.values().mean()
trueZeros = trueZerosPerSample.values().sum()
print(trueZeros)

# 5. Find the first two principal components
mat = RowMatrix(rows.values()) # drop sample IDs to do PCA

# mat is a ~ 10^6 x 28000 matrix

# We don't use mat.computePrincipalComponents since it does the computation on one machine
# and will run out of memory. Instead use SVD which is distributed for large matrices.
# pc = mat.computePrincipalComponents(2)

# center
# Don't center for the moment, since it is very memory intensive
#rowsn = rows.sample(False, 0.001) # downsample to ~1K rows so centering is OK
#means = rowsn.values().reduce(lambda a, b: Vectors.dense(a.toArray()) + Vectors.dense(b.toArray())) / rowsn.count()
#centeredRows = rowsn.values().map(lambda r: r - means)
#centeredRows.cache()
#print centeredRows.count() # TODO: very slow, no progress after 15 min (something to do with the map?)
#mat = RowMatrix(centeredRows)

# then run SVD
svd = mat.computeSVD(2, True)

u = svd.U.rows.collect()

# plot
xs = map(lambda v: v[0], u)
ys = map(lambda v: v[1], u)
import matplotlib.pyplot as plt
plt.scatter(xs, ys, marker=".", alpha=0.1)

# svd.U is 10^6 x 2
# svd.s is 2 x 2 diagnonal
# svd.V is 28000 x 2

# which gene is the most dominant?
v = svd.V.toArray()
import numpy as np
col1 = v[:, 0]
col2 = v[:, 1]
index1 = np.argmax(col1)
index2 = np.argmax(col2)
print index1, index2 # 'Malat1' gene

import math
def get_sorted(col):
  colabs = np.vectorize(lambda x : math.fabs(x))(col)
  return np.sort(colabs)[::-1]

# look at the drop off
col1sorted = get_sorted(col1)
col2sorted = get_sorted(col2)

