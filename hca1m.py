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
rows.count()

# 1. Find the number of measurements per sample
numMeasurementsPerSample = rows.mapValues(lambda vec : len(vec.values))
numMeasurementsPerSample.take(5)

# 2. Calculate the sparsity of the whole dataset (7%)
numMeasurementsPerSample.values().mean() / numFeatures

# 3. Find the number of true zeros (not NA) per sample (0)
trueZerosPerSample = rows.mapValues(lambda vec : sum(x == 0.0 for x in vec.values))
trueZerosPerSample.values().mean()
trueZerosPerSample.values().sum()

# 5. Find the first two principal components
mat = RowMatrix(rows.values()) # drop sample IDs to do PCA

# mat is a ~ 10^6 x 28000 matrix

# We don't use mat.computePrincipalComponents since it does the computation on one machine
# and will run out of memory. Instead use SVD which is distributed for large matrices.
# pc = mat.computePrincipalComponents(2)

# center
#means = rows.values().reduce(lambda a, b: Vectors.dense(a.toArray()) + Vectors.dense(b.toArray())) / rows.count()
#centeredRows = rows.values().map(lambda r: r - means)
#centeredRows.cache()
#print centeredRows.count() # TODO: very slow, no progress after 15 min (something to do with the map?)
#centeredMat = RowMatrix(centeredRows)

# then run SVD
svd = mat.computeSVD(2, True)

s = svd.U.rows.takeSample(False, 1000)
d = s[:1000]

# plot
xs = map(lambda v: v[0], d)
ys = map(lambda v: v[1], d)
import matplotlib.pyplot as plt
plt.scatter(xs, ys, marker=".", alpha=0.1)

# svd.U is 10^6 x 2
# svd.s is 2 x 2 diagnonal
# svd.V is 28000 x 2

# which is the strongest gene?
# multiple by (1 0 | 0 0) to get first column of V
# then do argmax to find which index has the largest value (in size)