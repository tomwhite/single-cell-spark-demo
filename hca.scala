// Start a Spark shell (`spark2-shell` on CDH) then run the following

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

import spark.implicits._

// Create a dataset. Each row is a sample, which consists of the sample ID and quantification values for features.
// The dataset is sparse, so features are referenced by index, and for each index a quantification value
// is specified. Features that are missing have no corresponding index.
// See https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.SparseVector
val numFeatures = 5
val data = Array(
  ("s1", Vectors.sparse(numFeatures, Array(1, 2, 3), Array(1.0, 0.0, 7.0))), // note explicit (true) zero!
  ("s2", Vectors.sparse(numFeatures, Array(0, 2, 3, 4), Array(2.0, 3.0, 4.0, 5.0))),
  ("s3", Vectors.sparse(numFeatures, Array(0, 3, 4), Array(4.0, 6.0, 7.0))))

// Turn the data into an RDD. Using parallelize is OK for datasets that fit in memory,
// but for larger datasets you would read from HDFS (e.g. from a tsv) and map into an RDD.
val rows: RDD[(String, Vector)] = sc.parallelize(data)

// Save the dataset in Parquet format in HDFS. To do this we first need to create a schema, which for this
// simple model is just the id and the indices and values arrays.
val schema = StructType(
    StructField("id", StringType, false) ::
    StructField("idx", ArrayType(IntegerType, false), false) ::
    StructField("quant", ArrayType(DoubleType, false), false) :: Nil)
// Then we map the data to Row objects (note we don't call `vec.toSparse` as it drops true zeros)...
// (see https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.Row)
val rowRDD = rows.map{ case (id, vec) => Row(id, vec.asInstanceOf[SparseVector].indices, vec.asInstanceOf[SparseVector].values) }
// ... so we can create a dataframe
// See https://spark.apache.org/docs/latest/sql-programming-guide.html#programmatically-specifying-the-schema
// See https://spark.apache.org/docs/latest/sql-programming-guide.html#parquet-files
val df = spark.createDataFrame(rowRDD, schema)
df.write.parquet("celldb")

// It's possible to look at the data in the Parquet file by using parquet-tools as follows:
// for f in $(hadoop fs -stat '%n' 'celldb/part-*'); do parquet-tools cat $(hdfs getconf -confKey fs.defaultFS)/user/$USER/celldb/$f; done
// You'll see that the true zero is stored, while the absence of a measurement is not.

// Load the data back in (this works from a new session too)
val df = spark.read.parquet("celldb")
val rows: RDD[(String, Vector)] = df.rdd.map(row => (row.getString(0), Vectors.sparse(numFeatures, row.getAs[Seq[Int]](1).toArray, row.getAs[Seq[Double]](2).toArray)))
rows.collect

// Now let's do some simple queries

// 1. Find the number of measurements per sample
val numMeasurementsPerSample = rows.mapValues(_.asInstanceOf[SparseVector].numActives)
numMeasurementsPerSample.collect // note that the first sample (s1) has 3 measurements, even though one is zero

// 2. Calculate the sparsity of the whole dataset
numMeasurementsPerSample.values.mean / numFeatures

// 3. Find the number of true zeros (not NA) per sample
val trueZerosPerSample = rows.mapValues(vec => {
  var count = 0
  vec.asInstanceOf[SparseVector].foreachActive((i,v) => if (v == 0.0) count += 1)
  count
})
trueZerosPerSample.collect

// 4. Project out features 0 and 2
val project = rows.mapValues(vec => Vectors.dense(vec(0), vec(2)))
project.collect

// 5. Find the first two principal components
// See https://spark.apache.org/docs/latest/mllib-dimensionality-reduction.html#principal-component-analysis-pca
// See https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.distributed.RowMatrix
val mat: RowMatrix = new RowMatrix(rows.values) // drop sample IDs to do PCA
val pc: Matrix = mat.computePrincipalComponents(2)
val projected: RowMatrix = mat.multiply(pc)
val projectedWithSampleIds = rows.keys.zip(projected.rows) // add back sample IDs; note can only call zip because projected has same partitioning and #rows per partition
projectedWithSampleIds.collect
