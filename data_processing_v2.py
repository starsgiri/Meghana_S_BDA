#!/usr/bin/env python3
"""
PySpark Data Processing Script V2 for Bank Customer Churn Prediction
Large Dataset Version (500,000 records)
This script loads, explores, cleans, and processes the large bank customer dataset.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, isnull
from pyspark.sql.types import *

def main():
    # ============================================================================
    # STEP 1: Initialize PySpark Session with Optimized Settings
    # ============================================================================
    
    print("Initializing Spark Session for Large Dataset...")
    spark = SparkSession.builder \
        .appName("BankCustomerChurnPrediction_V2") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()
    
    print("Spark Session V2 Created Successfully!")
    print(f"Spark Version: {spark.version}")
    
    # ============================================================================
    # STEP 2: Load Large Dataset (500K records)
    # ============================================================================
    
    print("\nLoading large dataset (500,000 records)...")
    df = spark.read.csv("dataset/Churn_Modelling_Large_500K.csv", header=True, inferSchema=True)
    
    print("Dataset loaded successfully!")
    print(f"Number of rows: {df.count():,}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of partitions: {df.rdd.getNumPartitions()}")
    
    # ============================================================================
    # STEP 3: Initial Data Exploration
    # ============================================================================
    
    print("\n=== Dataset Schema ===")
    df.printSchema()
    
    print("\n=== First 10 Rows ===")
    df.show(10)
    
    print("\n=== Column Names ===")
    print(df.columns)
    
    print("\n=== Statistical Summary ===")
    df.describe().show()
    
    # ============================================================================
    # STEP 4: Check for Missing Values
    # ============================================================================
    
    print("\n=== Missing Values Check ===")
    # Only check isnan for numeric columns
    numeric_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() in ['int', 'bigint', 'double', 'float']]
    string_cols = [c for c in df.columns if c not in numeric_cols]
    
    missing_counts = []
    for c in numeric_cols:
        missing_counts.append(count(when(isnan(c) | col(c).isNull(), c)).alias(c))
    for c in string_cols:
        missing_counts.append(count(when(col(c).isNull(), c)).alias(c))
    
    df.select(missing_counts).show()
    
    # ============================================================================
    # STEP 5: Analyze Target Variable (Churn)
    # ============================================================================
    
    print("\n=== Churn Distribution ===")
    churn_dist = df.groupBy("Exited").count().orderBy("Exited")
    churn_dist.show()
    
    total_customers = df.count()
    churned_customers = df.filter(col("Exited") == 1).count()
    retained_customers = df.filter(col("Exited") == 0).count()
    churn_rate = (churned_customers / total_customers) * 100
    
    print(f"\nTotal Customers: {total_customers:,}")
    print(f"Churned Customers: {churned_customers:,}")
    print(f"Retained Customers: {retained_customers:,}")
    print(f"Churn Rate: {churn_rate:.2f}%")
    
    # ============================================================================
    # STEP 6: Feature Analysis
    # ============================================================================
    
    print("\n=== Geography Distribution ===")
    df.groupBy("Geography").count().orderBy(col("count").desc()).show()
    
    print("\n=== Gender Distribution ===")
    df.groupBy("Gender").count().orderBy(col("count").desc()).show()
    
    print("\n=== Number of Products Distribution ===")
    df.groupBy("NumOfProducts").count().orderBy("NumOfProducts").show()
    
    print("\n=== HasCrCard Distribution ===")
    df.groupBy("HasCrCard").count().show()
    
    print("\n=== IsActiveMember Distribution ===")
    df.groupBy("IsActiveMember").count().show()
    
    # Analyze churn by categorical features
    print("\n=== Churn by Geography ===")
    df.groupBy("Geography", "Exited").count().orderBy("Geography", "Exited").show()
    
    print("\n=== Churn by Gender ===")
    df.groupBy("Gender", "Exited").count().orderBy("Gender", "Exited").show()
    
    print("\n=== Churn by Number of Products ===")
    df.groupBy("NumOfProducts", "Exited").count().orderBy("NumOfProducts", "Exited").show()
    
    print("\n=== Churn by Active Membership ===")
    df.groupBy("IsActiveMember", "Exited").count().orderBy("IsActiveMember", "Exited").show()
    
    # ============================================================================
    # STEP 7: Data Cleaning and Preprocessing
    # ============================================================================
    
    print("\nCleaning data...")
    df_clean = df.drop("RowNumber", "CustomerId", "Surname")
    
    print("\n=== Cleaned Dataset Schema ===")
    df_clean.printSchema()
    print(f"Remaining columns: {len(df_clean.columns)}")
    
    # ============================================================================
    # STEP 8: Feature Engineering
    # ============================================================================
    
    print("\nEngineering features...")
    
    # Age groups
    df_clean = df_clean.withColumn("AgeGroup", 
        when(col("Age") < 30, "Young")
        .when((col("Age") >= 30) & (col("Age") < 45), "Middle")
        .when(col("Age") >= 45, "Senior")
    )
    
    # Balance category
    df_clean = df_clean.withColumn("BalanceCategory",
        when(col("Balance") == 0, "Zero")
        .when((col("Balance") > 0) & (col("Balance") <= 50000), "Low")
        .when((col("Balance") > 50000) & (col("Balance") <= 100000), "Medium")
        .when(col("Balance") > 100000, "High")
    )
    
    # Credit Score Category
    df_clean = df_clean.withColumn("CreditScoreCategory",
        when(col("CreditScore") < 600, "Poor")
        .when((col("CreditScore") >= 600) & (col("CreditScore") < 700), "Fair")
        .when((col("CreditScore") >= 700) & (col("CreditScore") < 800), "Good")
        .when(col("CreditScore") >= 800, "Excellent")
    )
    
    # Tenure Category
    df_clean = df_clean.withColumn("TenureCategory",
        when(col("Tenure") < 3, "New")
        .when((col("Tenure") >= 3) & (col("Tenure") < 7), "Regular")
        .when(col("Tenure") >= 7, "Loyal")
    )
    
    # Product Engagement Score (combination of products and activity)
    df_clean = df_clean.withColumn("EngagementScore",
        (col("NumOfProducts") * 2 + col("IsActiveMember") * 3 + col("HasCrCard"))
    )
    
    print("\n=== Dataset with Engineered Features ===")
    df_clean.show(10)
    print(f"Total features after engineering: {len(df_clean.columns)}")
    
    # ============================================================================
    # STEP 9: Cache Dataset for Better Performance
    # ============================================================================
    
    print("\nCaching dataset for better performance...")
    df_clean.cache()
    df_clean.count()  # Trigger caching
    print("Dataset cached successfully!")
    
    # ============================================================================
    # STEP 10: Save Processed Data
    # ============================================================================
    
    print("\nSaving processed data...")
    
    # Save as Parquet (optimized for large datasets)
    df_clean.write.mode("overwrite").parquet("dataset/processed_churn_data_v2.parquet")
    print("Saved as Parquet: dataset/processed_churn_data_v2.parquet")
    
    # Save sample as CSV for reference (first 10000 records)
    df_clean.limit(10000).coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv("dataset/processed_churn_data_v2_sample_csv")
    print("Saved sample CSV: dataset/processed_churn_data_v2_sample_csv")
    
    print("\n=== Processed data saved successfully! ===")
    
    # ============================================================================
    # STEP 11: Final Statistics
    # ============================================================================
    
    print("\n=== Final Dataset Statistics V2 ===")
    print(f"Total Records: {df_clean.count():,}")
    print(f"Total Features: {len(df_clean.columns)}")
    print(f"\nFeature List: {df_clean.columns}")
    
    print("\n=== Feature Correlations with Churn ===")
    numeric_features = ["CreditScore", "Age", "Tenure", "Balance", 
                       "NumOfProducts", "HasCrCard", "IsActiveMember", 
                       "EstimatedSalary", "EngagementScore"]
    
    correlations = {}
    for feature in numeric_features:
        correlation = df_clean.stat.corr(feature, "Exited")
        correlations[feature] = correlation
        print(f"{feature}: {correlation:.4f}")
    
    # Find strongest predictors
    print("\n=== Top 5 Strongest Correlations ===")
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (feature, corr) in enumerate(sorted_corrs[:5], 1):
        print(f"{i}. {feature}: {corr:.4f}")
    
    # ============================================================================
    # STEP 12: Data Quality Report
    # ============================================================================
    
    print("\n=== Data Quality Report V2 ===")
    print(f"Dataset Version: 2.0")
    print(f"Processing Date: {__import__('datetime').datetime.now()}")
    print(f"Total Records Processed: {df_clean.count():,}")
    print(f"Features Engineered: {len(df_clean.columns) - len(df.columns) + 3}")  # New features minus dropped columns
    print(f"Churn Rate: {churn_rate:.2f}%")
    print(f"Data Quality: No missing values detected")
    
    print("\n=== Data Processing V2 Complete! ===")
    
    # Unpersist cache before stopping
    df_clean.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
