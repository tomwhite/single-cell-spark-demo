// https://github.com/LLNL/spark-hdf5
// spark-shell --jars spark-hdf5-0.0.4.jar
// After looking at the source code, it's clear it can only read from local files.
// So can run distributed on a cluster.

import gov.llnl.spark.hdf._

val df = sqlContext.read.hdf5("test1.h5", "/multi")
df.show