package com.spark.mylab

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

import scala.io.Source

/**
  * Created by baidu on 16/11/28.
  */
object Recommend {
  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[4]").setAppName("recommend")
    val sc = new SparkContext(conf)

    println("Begin rating file")
    // 装载数据集
    val text = sc.textFile("hdfs://Master:9000/moviedata/ratings.dat")
    val ratings = text.map {
      line =>
        val parts = line.split("::")
        (parts(3).toLong % 10, Rating(parts(0).toInt, parts(1).toInt, parts(2).toDouble))
    }


    val numPartitions = 4
    val training = ratings.filter(x => x._1 < 6) // 分组
      .values
      .repartition(numPartitions)
      .cache()
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .repartition(numPartitions)
      .cache()
    val test = ratings.filter(x => x._1 >= 8)
      .values
      .repartition(numPartitions)
      .cache()

    println("Finish data loading, train num: " + training.count()
      + " valid num: " + validation.count() + " test num: " + test.count())

    val ranks = List(8, 12)
    val lambdas = List(0.1, 10.0)
    val numIters = List(10, 20)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1

    println("Start train models with different parameters")
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda) // 注意参数的顺序
      val rmse = computeRmse(model, validation)
      println("RMSE=" + rmse + " for model with rank " + rank
        + " lambda " + lambda + " numIter " + numIter)
      if (rmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = rmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }

    }

    // 计算测试集合的结果
    val testRmse = computeRmse(bestModel.get, test)
    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + " and numIter = " + bestNumIter + " and its RMSE is " + testRmse)

    // 跟直接平均rating做对比
    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRmse = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating)).reduce(_ + _) / test.count)
    println("The best model improves the baseline by " + "%1.2f".format((baselineRmse - testRmse) / baselineRmse) + "%.")

    // 获取某一个用户的推荐结果
    val myratings = Source.fromFile("hdfs://Master:9000/moviedata/ratings.dat")
      .getLines()
      .map{
      line =>
        val parts = line.split("::")
        Rating(parts(0).toInt, parts(1).toInt, parts(2).toDouble)
    }

    // 读普通文件,用collect就可以
//    val myratings = sc.textFile("hdfs://Master:9000/moviedata/ratings.dat")
//      .collect()
//      .map {
//        line =>
//          val parts = line.split("::")
//          Rating(parts(0).toInt, parts(1).toInt, parts(2).toDouble)
//      }


    val ratedMovies = myratings.toSeq.map(_.product).toSet
    val myid = myratings.toSeq(0).user
    val movies = sc.textFile("hdfs://Master:9000/moviedata/movies.dat")
      .map {
        line =>
          val fields = line.split("::")
          (fields(0).toInt, fields(1))
      }
      .collect()
      .toMap

    val candid = movies.keys.filter(x => !ratedMovies.contains(x)).toSeq
    val cand = sc.parallelize(candid) // 注意这一行很神奇,其实是创建RDD

    val recommend = bestModel.get
      .predict(cand.map(x => (myid, x)))
      .collect()
      .sortBy(-_.rating) // 从大到小排序
      .take(10)

    var i = 1
    println("Movies recommeded for user " + myid)
    recommend.foreach{
      r =>
        println("%2d".format(i) + ": " + movies.get(r.product))
        i += 1
    }

    sc.stop()
    println("All elements done")

  }

  // 根据实际数据的均方根误差来判断效果
  def computeRmse(model:MatrixFactorizationModel, data:RDD[Rating]): Double = {
    val predict: RDD[Rating] = model.predict(data.map(x=>(x.user, x.product)))
    val comparePredict = predict.map(x=>((x.user, x.product), x.rating))
      .join(data.map(x=>((x.user, x.product), x.rating)))
      .values

    val n = predict.count()
    math.sqrt(comparePredict.map(x=>(x._1-x._2)*(x._1-x._2)).reduce(_+_)/n)
  }

}