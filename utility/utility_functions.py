import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def combine_data_from_file(file_path: str, sheet_names: list[str] = None, output_dir: str = 'processed_data') -> None:
    """
    Combines data from specified sheets in an Excel workbook and saves the result to a new Excel file.
    
    Parameters:
    - file_path (str): The path to the Excel workbook.
    - sheet_names (list[str], optional): List of sheet names to combine. If None, all sheets are combined.
    - output_dir (str, optional): Directory to save the combined Excel file. Defaults to 'processed_data'.
    """
    # Determine the appropriate engine for reading the file
    engine = 'pyxlsb' if file_path.endswith('.xlsb') else 'openpyxl'

    # Load the workbook and read the sheets
    with pd.ExcelFile(file_path, engine=engine) as xls:
        # If no sheet names are provided, use all sheets
        if sheet_names is None:
            sheet_names = xls.sheet_names

        # Read and combine the specified sheets
        combined_data = pd.concat([pd.read_excel(xls, sheet_name=sheet) for sheet in sheet_names], ignore_index=True)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the combined data to a new Excel file
    output_path = os.path.join(output_dir, 'combined_data.csv')
    combined_data.to_csv(output_path, index=False)

    print(f"Data combined and saved successfully to {output_path}")


def dist_plot(rfm_table: pd.DataFrame, target: str, kmeans: bool = False) -> None:
    """
    Plot the RFM distribution for the target column (Recency, Frequency, Monetary).
    
    Parameters:
    - rfm_table (pd.DataFrame): DataFrame containing Recency, Frequency, and Monetary scores.
    - target (str): Specify which score to plot (e.g., 'Recency', 'Frequency', or 'Monetary').
    - kmeans (bool): If True, use the 'Segment_Kmeans' column for segment grouping. Default is False.
    """
    
    # Choose the correct segment column based on the kmeans flag
    segment_col = 'Cluster' if kmeans else 'Segment'
    
    # Get unique segments and assign distinct colors to each
    segments = rfm_table[segment_col].unique()
    color_palette = sns.color_palette('CMRmap', len(segments))
    color_map = dict(zip(segments, color_palette))
    
    plt.figure(figsize=(15, 10))
    
    # Create the scatter plot with different colors for each segment
    for segment in segments:
        segment_data = rfm_table[rfm_table[segment_col] == segment]
        plt.scatter(segment_data['Customer ID'], 
                    segment_data[target], 
                    label=segment, 
                    color=color_map[segment], 
                    alpha=0.9, 
                    edgecolor='w', 
                    linewidth=0.9)
    
    # Add title and labels
    plt.title(f'Customer {target} Distribution by Segment', fontsize=16)
    plt.xlabel('Customer ID', fontsize=12)
    plt.ylabel(target, fontsize=12)
    
    # Add a legend
    plt.legend(title='Customer Segments', loc='upper right')
    
    # Customize axis limits and add grid for better readability
    plt.ylim(0, rfm_table[target].max() * 1.1)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# obtain correlation matrix
def cormat(df):

    df = df.drop(['Customer ID'],axis=1)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    cmat = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(cmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=df[numeric_columns].columns, 
                 xticklabels=df[numeric_columns].columns, 
                 cmap="Spectral_r")
    plt.show()
    
# Exploratory Data Analysis function
def perform_eda(df):
    
    # Show the descriptive details about the dataset
    print("".rjust(38, '='))
    print(f"The First 5 Rows of the Dataset")
    print("".rjust(38, '='))
    print(df.head())
    print("".rjust(38, '='))
    print(f"The Last 5 Rows of the Dataset")
    print("".rjust(38, '='))
    print(df.tail())
    print("".rjust(38, '='))
    print(f" The Dimension of the Dataset")
    print("".rjust(38, '='))
    print(df.shape)
    print("".rjust(38, '='))
    print(f" General information about the dataset")
    print("".rjust(38, '='))
    df.info()
    print("".rjust(38, '='))
    print(f" Number of Missing Data By Column")
    print("".rjust(38, '='))
    print(df.isna().sum())
    print("".rjust(38, '='))
    print(f"\n Number of Duplicated Rows")
    print("".rjust(38, '='))
    print(df.duplicated().sum())
    print("".rjust(38, '='))
    print(f"\n Number of Unique Values")
    print("".rjust(38, '='))
    print(df.nunique())
    print("".rjust(38, '='))
    print(f"\n Descriptive statistics:\n")
    print("".rjust(38, '='))
    print(df.describe(include=['float','int']))

# This function imputes missing values for numeric and categorical columns
def handle_missing_values(df):
    
    imputer = SimpleImputer(strategy='median')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df.loc[:,numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    imputer = SimpleImputer(strategy='most_frequent')
    categorical_columns = df.select_dtypes(include=['object']).columns
    df.loc[:,categorical_columns] = imputer.fit_transform(df[categorical_columns])
    return df

def drop_duplicate(dataframe):
    return dataframe.drop_duplicates()

def load_data(dataset_path):

    #Import the dataset into a dataframe
    df = pd.read_csv(dataset_path,low_memory=False)
    
    return df

# A function for removing outliers
def remove_outliers_zscore(df, target_columns,threshold=3):
    """
    Removes outliers from a DataFrame based on z-scores for a given column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_columns (list): The columns to evaluate for outliers.
        threshold (float): The z-score threshold for identifying outliers.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    
    # Calculate the z-scores for the columns
    z_scores = np.abs(stats.zscore(df[target_columns]))
    
    # Filter out rows where the z-score is greater than the threshold
    filtered_entries = (z_scores < threshold).all(axis=1)

    # Combine the filtered X with the target column and return
    df_cleaned = df[filtered_entries]

    return df_cleaned
