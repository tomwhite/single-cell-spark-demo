# Start a Spark shell (`pyspark2` on CDH) then run the following

from pyspark.ml.linalg import Vectors
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
a = rowRDD.collect()
# ... so we can create a dataframe
# See https://spark.apache.org/docs/latest/sql-programming-guide.html#programmatically-specifying-the-schema
# See https://spark.apache.org/docs/latest/sql-programming-guide.html#parquet-files
df = spark.createDataFrame(rowRDD, schema)
df.write.parquet("celldbpy")

# It's possible to look at the data in the Parquet file by using parquet-tools as follows:
# for f in $(hadoop fs -stat '%n' 'celldbpy/part-*'); do parquet-tools cat $(hdfs getconf -confKey fs.defaultFS)/user/$USER/celldbpy/$f; done
# You'll see that the true zero is stored, while the absence of a measurement is not.