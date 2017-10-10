# Start a Spark shell (`pyspark2` on CDH) then run the following

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

# Create a dataset. Each row is a sample, which consists of the sample ID and
# quantification values for features.
# The dataset is sparse, so features are referenced by index, and for each index a
# quantification value
# is specified. Features that are missing have no corresponding index.
# See https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.SparseVector
numFeatures = 5
data = [
    ("s1", Vectors.sparse(numFeatures, {1: 1.0, 2: 0.0, 3: 7.0})), # note explicit (true) zero!
    ("s2", Vectors.sparse(numFeatures, {0: 2.0, 2: 3.0, 3: 4.0, 4: 5.0})),
    ("s3", Vectors.sparse(numFeatures, {0: 4.0, 3: 6.0, 4: 7.0}))
]

# Turn the data into an RDD. Using parallelize is OK for datasets that fit in memory,
# but for larger datasets you would read from HDFS (e.g. from a tsv) and map into an RDD.
rows = sc.parallelize(data)

# Save the dataset in Parquet format in HDFS. To do this we first need to create a
# schema, which for this simple model is just the id and the indices and values arrays.
schema = StructType([StructField("id", StringType(), False),
    StructField("idx", ArrayType(IntegerType(), False), False),
    StructField("quant", ArrayType(DoubleType(), False), False)])
# Then we map the data to Row objects (tuples in Python). Note use of tolist() to
# convert from numpy arrays to regular Python lists
rowRDD = rows.map(lambda (id, vec): (id, vec.indices.tolist(), vec.values.tolist()))
# ... so we can create a dataframe
# See https://spark.apache.org/docs/latest/sql-programming-guide.html#programmatically-specifying-the-schema
# See https://spark.apache.org/docs/latest/sql-programming-guide.html#parquet-files
df = spark.createDataFrame(rowRDD, schema)
df.write.parquet("celldb")

# It's possible to look at the data in the Parquet file by using parquet-tools as follows:
# for f in $(hadoop fs -stat '%n' 'celldb/part-*'); do parquet-tools cat $(hdfs getconf -confKey fs.defaultFS)/user/$USER/celldb/$f; done
# You'll see that the true zero is stored, while the absence of a measurement is not.

# Load the data back in (this works from a new session too)
df = spark.read.parquet("celldb")
rows = df.rdd.map(lambda (id, indices, values): (id, Vectors.sparse(numFeatures, indices, values )))
rows.collect()

# Now let's do some simple queries

# 1. Find the number of measurements per sample
numMeasurementsPerSample = rows.mapValues(lambda vec : len(vec.values))
numMeasurementsPerSample.collect() # note that the first sample (s1) has 3 measurements, even though one is zero

# 2. Calculate the sparsity of the whole dataset
numMeasurementsPerSample.values().mean() / numFeatures

# 3. Find the number of true zeros (not NA) per sample
trueZerosPerSample = rows.mapValues(lambda vec : sum(x == 0.0 for x in vec.values))
trueZerosPerSample.collect()

# 4. Project out features 0 and 2
project = rows.mapValues(lambda vec: Vectors.dense(vec[0], vec[2]))
project.collect()

# 5. Find the first two principal components
# See https://spark.apache.org/docs/latest/mllib-dimensionality-reduction.html#principal-component-analysis-pca
# See https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.distributed.RowMatrix
mat = RowMatrix(rows.values()) # drop sample IDs to do PCA
pc = mat.computePrincipalComponents(2)
projected = mat.multiply(pc)
projectedWithSampleIds = rows.keys().zip(projected.rows) # add back sample IDs; note can only call zip because projected has same partitioning and #rows per partition
projectedWithSampleIds.collect()