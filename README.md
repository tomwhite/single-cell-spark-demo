## Experiments with Single Cell data using Spark

This repo contains some experiments on Single Cell data from 10x Genomics using Apache Spark.

### Data Model

Before downloading a large dataset, it's worth looking at the sparse data model used. This is demonstrated in `hca.py`, which uses a few rows of toy data as a demonstration.

This file can be run on a Spark cluster using `pyspark2`, or in a CDSW session.

The `hca.scala` file is the equivalent in Scala.

### Data

The data can be downloaded from the [10x Genomics website](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons) (registration required).

The data is in HDF5 format, which need to be converted to a format that Spark can read efficiently. I tried a few ways of doing this - none were particularly effective, due to library issues, memory issues or lack of parallelism. The best way that worked is in `hdf_to_parquet.py`, although it is slow since it reads from local files. Ideally I would find a way that can process in parallel on a Hadoop filesystem.

The output of this process is a set of files in the `celldb1m` directory in HDFS. This contains 1 million rows of data.

### Processing

Use the `hca1m.py` file to run the analysis on the 1 million rows data.

Note that doing PCA does not work on the full dataset when the data is centered, since the matrix is no longer sparse, so it is very computationally intensive.
