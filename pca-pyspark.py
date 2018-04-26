# https://blog.dominodatalab.com/pca-on-very-large-neuroimaging-datasets-using-pyspark/

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("pca") \
    .getOrCreate()
sc = spark.sparkContext

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

data = [
    (Vectors.dense([0.0, 1.0, 0.0, 7.0, 0.0]),),
    (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
    (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)
]
data

df = spark.createDataFrame(data, ["features"])

pca_extracted = PCA(k=2, inputCol="features", outputCol="pca_features")

model = pca_extracted.fit(df)
features = model.transform(df)

# Components (matrix V)
identity_input = [
    (Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0),),
    (Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0),),
    (Vectors.dense(0.0, 0.0, 1.0, 0.0, 0.0),),
    (Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0),),
    (Vectors.dense(0.0, 0.0, 0.0, 0.0, 1.0),)
]
df_identity = spark.createDataFrame(identity_input, ["features"])
identity_features = model.transform(df_identity)
v = identity_features.select("pca_features").collect()
v # v is 5 x 2

# Representation in 2-D (matrix Z)
z = features.select("pca_features").collect()
z # z is 3 x 2