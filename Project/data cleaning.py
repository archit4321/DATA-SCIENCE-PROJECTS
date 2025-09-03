import pandas as pd
import numpy as np

def data_preprocessing_tool(df, missing_value_strategy='impute_mean', outlier_method='zscore', zscore_threshold=3.0, iqr_factor=1.5):
    """
    A comprehensive data cleaning and preprocessing tool.

    Args:
        df (pd.DataFrame): The input pandas DataFrame to be cleaned.
        missing_value_strategy (str): Strategy for handling missing values.
            - 'impute_mean': Fills missing values with the mean of the column.
            - 'impute_median': Fills missing values with the median of the column.
            - 'impute_mode': Fills missing values with the mode of the column.
            - 'drop': Drops rows with any missing values.
        outlier_method (str): Method for detecting outliers.
            - 'zscore': Uses the Z-score method.
            - 'iqr': Uses the Interquartile Range (IQR) method.
        zscore_threshold (float): The threshold for Z-score outlier detection.
        iqr_factor (float): The factor for IQR outlier detection.

    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    print("Starting data preprocessing...")

    # --- Step 1: Duplicate Removal ---
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicate_rows_removed = initial_rows - len(df)
    print(f"Removed {duplicate_rows_removed} duplicate rows.")

    # --- Step 2: Handling Missing Values ---
    # Create a copy to avoid modifying the original DataFrame in place when using `drop`
    df_cleaned = df.copy()
    
    missing_before = df_cleaned.isnull().sum().sum()
    if missing_value_strategy == 'drop':
        df_cleaned.dropna(inplace=True)
        print(f"Dropped rows with missing values. {missing_before} missing values were present.")
    elif missing_value_strategy in ['impute_mean', 'impute_median', 'impute_mode']:
        for col in df_cleaned.columns:
            # We only impute for numeric columns
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                if missing_value_strategy == 'impute_mean':
                    fill_value = df_cleaned[col].mean()
                elif missing_value_strategy == 'impute_median':
                    fill_value = df_cleaned[col].median()
                else: # impute_mode
                    fill_value = df_cleaned[col].mode()[0]
                df_cleaned[col].fillna(fill_value, inplace=True)
                
        missing_after = df_cleaned.isnull().sum().sum()
        print(f"Imputed missing values. {missing_before - missing_after} values were imputed.")
    else:
        print(f"Invalid missing value strategy: '{missing_value_strategy}'. No missing value handling performed.")

    # --- Step 3: Outlier Detection (and removal) ---
    outliers_removed = 0
    df_final = df_cleaned.copy()
    
    # We only detect outliers in numeric columns
    numeric_cols = df_final.select_dtypes(include=np.number).columns
    
    if outlier_method == 'zscore':
        for col in numeric_cols:
            mean = df_final[col].mean()
            std = df_final[col].std()
            if std == 0:  # Avoid division by zero
                continue
            df_final['zscore'] = np.abs((df_final[col] - mean) / std)
            outliers_found = df_final[df_final['zscore'] > zscore_threshold]
            outliers_removed += len(outliers_found)
            df_final = df_final[df_final['zscore'] <= zscore_threshold].drop('zscore', axis=1)

    elif outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = df_final[col].quantile(0.25)
            Q3 = df_final[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            outliers_found = df_final[(df_final[col] < lower_bound) | (df_final[col] > upper_bound)]
            outliers_removed += len(outliers_found)
            df_final = df_final[(df_final[col] >= lower_bound) & (df_final[col] <= upper_bound)]
    else:
        print(f"Invalid outlier method: '{outlier_method}'. No outlier detection performed.")

    print(f"Removed {outliers_removed} outliers.")
    print("Data preprocessing complete.")
    return df_final

# --- Example Usage ---
if __name__ == "__main__":
    # Create a sample DataFrame with missing values, duplicates, and outliers
    sample_data = {
        'column_A': [10, 20, 30, 40, 50, 10, 60, np.nan],
        'column_B': [1, 2, 3, 4, 5, 1, 6, 100],  # 100 is an outlier
        'column_C': ['a', 'b', 'c', 'd', 'e', 'a', 'f', 'g'],
    }
    sample_df = pd.DataFrame(sample_data)

    print("--- Original DataFrame ---")
    print(sample_df)
    print("\n--------------------------\n")
    
    # Example 1: Use the default parameters (impute mean, z-score outlier)
    cleaned_df_1 = data_preprocessing_tool(sample_df)
    print("\n--- Cleaned DataFrame (Default Parameters) ---")
    print(cleaned_df_1)

    print("\n\n--------------------------\n")
    
    # Example 2: Drop missing values and use IQR for outliers
    cleaned_df_2 = data_preprocessing_tool(sample_df, missing_value_strategy='drop', outlier_method='iqr')
    print("\n--- Cleaned DataFrame (Drop missing values, IQR outlier) ---")
    print(cleaned_df_2)
