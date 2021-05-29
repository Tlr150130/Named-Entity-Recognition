// Databricks notebook source
// Prior libraries need to be installed on the cluster before the code can be ran as of 5/28/2021:
// 1. Create a cluster if you don’t have one already

// 2. On a new cluster or existing one you need to add the following to the Advanced Options -> Spark tab:
//     spark.kryoserializer.buffer.max 2000M
//     spark.serializer org.apache.spark.serializer.KryoSerializer

// 3. In Libraries tab inside your cluster you need to follow these steps:
//     3.1. Install New -> PyPI -> spark-nlp -> Install
//     3.2. Install New -> Maven -> Coordinates -> com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.3 -> Install

// Now you can attach your notebook to the cluster and use Spark NLP!

// COMMAND ----------

// Libraries
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._

// COMMAND ----------

// premade pipeline from johnsnow labs initialization
val pipeline = PretrainedPipeline("recognize_entities_dl", lang="en")

// COMMAND ----------

// Load and prepare text (Hounds of the Baskervilles)
val input = sc.textFile("/FileStore/tables/houndofthebaskervilles.txt")

// split at punctuations
val words = input.flatMap(line => line.split("""\W+"""))
 
// transfer to a DF of arrays
val workable = words.toDF("text")

// transform data
val results = pipeline.transform(workable)

// extract only the named entity from the dataframe
val interest = results.select("entities.result")

// transform dataframe of arrays into dataframe of strings
val df = interest.map(f => {
  val result = f.getList(0).toArray.mkString(",")
  (result)
}).filter("value != ''")

// get count and sort
val result = df
  .rdd.map(x => (x,1)) // turn each row into a key and singular count
  .reduceByKey((count1, count2) => (count1+count2)) // aggregate counts by key
  .sortBy(-_._2) // sort by descending count
  .toDF("Entity", "Count") // turn into a dataframe with labeled columns

// COMMAND ----------

val df = interest.map(f => {
  val result = f.getList(0).toArray.mkString(",")
  (result)
}).filter("value != ''")

// COMMAND ----------

val result = df.rdd.map(x => (x,1)).reduceByKey((count1, count2) => (count1+count2)).sortBy(-_._2).toDF("Entity", "Count")

// COMMAND ----------

display(result)

// COMMAND ----------


