#!/usr/bin/env python3
"""
Generate synthetic bank customer churn dataset with 500,000 records
Combines and augments existing dataset to create larger training set
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_large_dataset():
    print("=" * 80)
    print("GENERATING LARGE DATASET (500,000 RECORDS)")
    print("=" * 80)
    
    # Load existing datasets
    print("\nLoading existing datasets...")
    df1 = pd.read_csv("dataset/Churn_Modelling.csv")
    df2 = pd.read_csv("dataset/botswana_bank_customer_churn.csv")
    
    print(f"Dataset 1 rows: {len(df1)}")
    print(f"Dataset 2 rows: {len(df2)}")
    
    # Standardize columns for dataset 2 if needed
    print("\nStandardizing dataset structure...")
    
    # Check columns
    print(f"\nDataset 1 columns: {df1.columns.tolist()}")
    print(f"Dataset 2 columns: {df2.columns.tolist()}")
    
    # Combine datasets
    # If dataset 2 has similar structure, combine them
    # Otherwise, use dataset 1 and augment it
    
    # Use the first dataset as base
    base_df = df1.copy()
    
    # Calculate how many times to replicate
    target_size = 500000
    current_size = len(base_df)
    
    print(f"\nTarget size: {target_size:,} records")
    print(f"Current size: {current_size:,} records")
    
    # Create augmented dataset
    print("\nGenerating synthetic variations...")
    
    dfs_to_concat = [base_df]
    
    # Replicate with slight variations
    np.random.seed(42)
    
    replications_needed = (target_size // current_size) + 1
    print(f"Replications needed: {replications_needed}")
    
    for i in range(1, replications_needed):
        print(f"Creating variation {i}...")
        
        df_copy = base_df.copy()
        
        # Add noise to numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['RowNumber', 'CustomerId', 'Exited']:
                # Add random noise (±5%)
                noise = np.random.normal(1, 0.05, len(df_copy))
                df_copy[col] = (df_copy[col] * noise).astype(df_copy[col].dtype)
        
        # Modify CustomerId to make unique
        if 'CustomerId' in df_copy.columns:
            df_copy['CustomerId'] = df_copy['CustomerId'] + (i * 20000000)
        
        # Modify RowNumber
        if 'RowNumber' in df_copy.columns:
            df_copy['RowNumber'] = df_copy['RowNumber'] + (i * len(base_df))
        
        dfs_to_concat.append(df_copy)
    
    # Combine all variations
    print("\nCombining all variations...")
    large_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Trim to exact target size
    large_df = large_df.head(target_size)
    
    # Reset row numbers
    if 'RowNumber' in large_df.columns:
        large_df['RowNumber'] = range(1, len(large_df) + 1)
    
    print(f"\nFinal dataset size: {len(large_df):,} records")
    print(f"Dataset shape: {large_df.shape}")
    
    # Ensure numeric columns are within valid ranges
    print("\nAdjusting numeric ranges...")
    
    if 'CreditScore' in large_df.columns:
        large_df['CreditScore'] = large_df['CreditScore'].clip(300, 850).astype(int)
    
    if 'Age' in large_df.columns:
        large_df['Age'] = large_df['Age'].clip(18, 100).astype(int)
    
    if 'Tenure' in large_df.columns:
        large_df['Tenure'] = large_df['Tenure'].clip(0, 10).astype(int)
    
    if 'Balance' in large_df.columns:
        large_df['Balance'] = large_df['Balance'].clip(0, 250000).round(2)
    
    if 'NumOfProducts' in large_df.columns:
        large_df['NumOfProducts'] = large_df['NumOfProducts'].clip(1, 4).astype(int)
    
    if 'EstimatedSalary' in large_df.columns:
        large_df['EstimatedSalary'] = large_df['EstimatedSalary'].clip(0, 200000).round(2)
    
    # Save the large dataset
    output_file = "dataset/Churn_Modelling_Large_500K.csv"
    print(f"\nSaving dataset to {output_file}...")
    large_df.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"File saved successfully! Size: {file_size:.2f} MB")
    
    # Display statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal Records: {len(large_df):,}")
    print(f"Total Features: {len(large_df.columns)}")
    print(f"\nColumns: {large_df.columns.tolist()}")
    
    print("\nChurn Distribution:")
    if 'Exited' in large_df.columns:
        churn_dist = large_df['Exited'].value_counts()
        print(churn_dist)
        churn_rate = (churn_dist.get(1, 0) / len(large_df)) * 100
        print(f"Churn Rate: {churn_rate:.2f}%")
    
    print("\nFirst few records:")
    print(large_df.head())
    
    print("\nData types:")
    print(large_df.dtypes)
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    
    return output_file

if __name__ == "__main__":
    output_file = generate_large_dataset()
    print(f"\nDataset ready at: {output_file}")
