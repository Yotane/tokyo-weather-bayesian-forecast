import os
import sys
import numpy as np
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, hour, dayofweek, month, lag,
    avg, stddev, sum as spark_sum, sqrt, pow as spark_pow,
    sin, cos, log1p, lit
)
from pyspark.sql.window import Window

# config: features to process
RAW_FEATURES = ["temperature_2m", "precipitation", "humidity", "pressure_msl", "wind_u", "wind_v"]
LAG_COLS = ["temperature_2m", "precipitation"]  # most predictive for short-horizon forecasting
ROLLING_MEAN_COLS = ["temperature_2m", "humidity", "pressure_msl"]
ROLLING_STD_COLS = ["temperature_2m", "pressure_msl"]
ROLLING_SUM_COLS = ["precipitation"]

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

        # defensive: check which expected columns actually exist
        available_cols = [c for c in RAW_FEATURES if c in df.columns]
        if not available_cols:
            raise ValueError(f"None of expected features {RAW_FEATURES} found in csv. Columns: {df.columns}")

        print(f"Processing features: {available_cols}")

        # parse timestamps with explicit format for robustness
        df = df.withColumn("timestamp", to_timestamp(col("time"), "yyyy-MM-dd'T'HH:mm")).drop("time")

        # extract cyclical time features
        df = df.withColumn("hour", hour("timestamp")) \
               .withColumn("day_of_week", dayofweek("timestamp")) \
               .withColumn("month", month("timestamp")) \
               .withColumn("is_weekend", col("day_of_week").isin([1, 7]).cast("int"))

        # add cyclical encodings to avoid boundary discontinuities (23->0, dec->jan)
        df = df.withColumn("hour_sin", sin(2 * lit(np.pi) * col("hour") / 24)) \
               .withColumn("hour_cos", cos(2 * lit(np.pi) * col("hour") / 24)) \
               .withColumn("month_sin", sin(2 * lit(np.pi) * col("month") / 12)) \
               .withColumn("month_cos", cos(2 * lit(np.pi) * col("month") / 12))

        # define window for time series operations (global order preserves cross-year continuity)
        w = Window.orderBy("timestamp")

        # create lag features
        for feat in LAG_COLS:
            if feat in available_cols:
                df = df.withColumn(f"{feat}_lag_1h", lag(feat, 1).over(w)) \
                       .withColumn(f"{feat}_lag_24h", lag(feat, 24).over(w))

        # compute rolling statistics over 24-hour windows (exclude current timestep to prevent leakage)
        for feat in ROLLING_MEAN_COLS:
            if feat in available_cols:
                df = df.withColumn(
                    f"{feat}_24h_mean",
                    avg(feat).over(w.rowsBetween(-24, -1))
                )

        for feat in ROLLING_STD_COLS:
            if feat in available_cols:
                df = df.withColumn(
                    f"{feat}_24h_std",
                    stddev(feat).over(w.rowsBetween(-24, -1))
                )

        for feat in ROLLING_SUM_COLS:
            if feat in available_cols:
                df = df.withColumn(
                    f"{feat}_24h_sum",
                    spark_sum(feat).over(w.rowsBetween(-24, -1))
                )

        # wind-specific derived features (if u/v components exist)
        if "wind_u" in available_cols and "wind_v" in available_cols:
            df = df.withColumn(
                "wind_speed",
                sqrt(spark_pow(col("wind_u"), 2) + spark_pow(col("wind_v"), 2))
            ).withColumn(
                "wind_speed_24h_mean",
                avg(
                    sqrt(spark_pow(col("wind_u"), 2) + spark_pow(col("wind_v"), 2))
                ).over(w.rowsBetween(-24, -1))
            )

        # add log transform for skewed precipitation data
        if "precipitation" in available_cols:
            df = df.withColumn("precipitation_log", log1p(col("precipitation")))

        # clean missing values (lag/rolling ops introduce nulls at start of series)
        df = df.dropna()

        # collect to pandas and write with pandas (bypasses winutils issues, fine for ~500k rows)
        pdf = df.toPandas()

        output_path = "data/tokyo_features.csv"
        pdf.to_csv(output_path, index=False)

        row_count = len(pdf)
        col_count = len(pdf.columns)

        print(f"ETL complete. Saved features to {output_path}")
        print(f"  Rows after cleaning: {row_count:,}")
        print(f"  Features generated: {col_count}")
        print(f"  Columns: {list(pdf.columns)}")

    except Exception as e:
        print(f"Error during etl process: {e}")
        raise e

    finally:
        spark.stop()

if __name__ == "__main__":
    main()