import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, month, lag, avg, stddev, sum as spark_sum
from pyspark.sql.window import Window

def main():
    # initialize spark session with increased driver memory to handle collect
    spark = SparkSession.builder \
        .appName("tokyo_weather_etl") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "localhost") \
        .config("spark.network.timeout", "600s") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        # load csv and parse timestamps
        df = spark.read.csv("data/tokyo_weather_1970_2024.csv", header=True, inferSchema=True)
        df = df.withColumn("timestamp", to_timestamp(col("time"))).drop("time")

        # extract cyclical time features
        df = df.withColumn("hour", hour("timestamp")) \
               .withColumn("day_of_week", dayofweek("timestamp")) \
               .withColumn("month", month("timestamp")) \
               .withColumn("is_weekend", col("day_of_week").isin([1, 7]).cast("int"))

        # define window for time series operations
        w = Window.orderBy("timestamp")

        # create lag features
        df = df.withColumn("temp_lag_1h", lag("temperature_2m", 1).over(w)) \
               .withColumn("temp_lag_24h", lag("temperature_2m", 24).over(w)) \
               .withColumn("precip_lag_1h", lag("precipitation", 1).over(w))

        # compute rolling statistics over 24-hour windows
        df = df.withColumn("temp_24h_mean", avg("temperature_2m").over(w.rowsBetween(-23, 0))) \
               .withColumn("temp_24h_std", stddev("temperature_2m").over(w.rowsBetween(-23, 0))) \
               .withColumn("precip_24h_sum", spark_sum("precipitation").over(w.rowsBetween(-23, 0)))

        # clean missing values
        df = df.dropna()
        
        # repartition by year for better pandas conversion performance
        df = df.repartition("year")
        
        # collect to pandas and write with pandas (bypasses winutils issues)
        pdf = df.toPandas()
        output_path = "data/tokyo_features.csv"
        pdf.to_csv(output_path, index=False)

        row_count = len(pdf)
        print(f"ETL complete. Saved features to {output_path}")
        print(f"Row count after cleaning: {row_count:,}")

    except Exception as e:
        print(f"Error during ETL process: {e}")
        raise e
    finally:
        spark.stop()

if __name__ == "__main__":
    main()