import pandas as pd
from fuzzywuzzy import fuzz
from collections import defaultdict
import numpy as np
import pickle as pkl
import logging

def process_bank_statements(dataframes):
    # Define standard column expectations
    EXPECTED_COLUMNS = {
        'Date': ['date', 'transaction date', 'posted date', 'value date'],
        'Description': ['description', 'narration', 'details', 'merchant', 'item', 'particulars', 'name', 'transaction'],
        'Credit': ['credit', 'cr', 'deposit', 'received', 'in', 'inflow'],
        'Debit': ['debit', 'dr', 'withdrawal', 'out', 'outflow'],
        'Balance': ['balance', 'current balance', 'available']
    }

    # Separate dataframes by size
    small_dfs = [df for df in dataframes if len(df.columns) < 5]
    processable_dfs = [df for df in dataframes if len(df.columns) >= 5]

    # Step 1: Identify labeled dataframes
    labeled_groups = dict()
    labeled_cols = dict()
    unlabeled_dfs = []
    
    for df in processable_dfs:
        if is_labeled_dataframe(df, EXPECTED_COLUMNS):
            structure = len(df.columns)
            if structure not in labeled_groups:
                labeled_groups[structure] = pd.DataFrame(columns=df.columns)
                labeled_cols[structure] = extract_column_names(df, EXPECTED_COLUMNS)

            labeled_groups[structure] = pd.concat([labeled_groups[structure], df], ignore_index=True)
        else:
            unlabeled_dfs.append(df)

    # Step 2: Match unlabeled to labeled
    processed = []
    for structure, labeled_df in labeled_groups.items():
        # Find all unlabeled with matching structure
        matching_unlabeled = [df for df in unlabeled_dfs if len(df.columns) == structure]
        
        labeled_df = process_labeled(labeled_df, labeled_cols[structure])
            
        # Process and merge matching unlabeled
        for unlabeled_df in matching_unlabeled:
            processed_unlabeled = process_unlabeled(unlabeled_df, labeled_cols[structure])
            labeled_df = pd.concat([labeled_df, processed_unlabeled], ignore_index=True)
        
        processed.append(labeled_df)
    
    return processed, small_dfs

def is_labeled_dataframe(df: pd.DataFrame, expected_columns: list) -> bool:
    try:
        # Validate DataFrame is not empty
        if df.empty:
            logging.warning("Empty DataFrame received")
            return False

        # Check column types and names
        if not all(isinstance(col, str) for col in df.columns):
            df.columns = df.columns.astype(str)

        # Convert first row to lowercase strings with null handling
        first_row = df.iloc[0].fillna('').astype(str).str.lower().values
        matches = 0
        
        for cell in first_row:
            for col_type, keywords in expected_columns.items():
                if any(fuzz.partial_ratio(cell, kw) >= 75 for kw in keywords):
                    matches += 1
                    break
                    
        return matches >= 3  # At least 3 columns match expected labels
    except Exception as e:
        logging.error(f"Error validating DataFrame: {str(e)}", exc_info=True)
        return False

def extract_column_names(df, expected_columns):
    first_row = df.iloc[0].astype(str).str.strip()
    
    column_mapping = {} # map index: standardized col, or unknown
    for col_type, keywords in expected_columns.items():
        best_match = None
        best_score = 0
        for idx, cell in enumerate(first_row):
            if idx in column_mapping:
                continue
            for kw in keywords:
                score = fuzz.partial_ratio(cell.lower(), kw)
                if score > best_score:
                    best_score = score
                    best_match = idx

        column_mapping[best_match] = col_type

    for i in range(len(first_row)):
        if i not in column_mapping:
            column_mapping[i] = 'unknown'
    
    return column_mapping

def process_labeled(df, column_mapping):
    processed = df.iloc[1:].reset_index(drop=True)
    processed.columns = [column_mapping.get(i, 'Unknown') for i in range(len(df.columns))]
    processed = processed.drop(columns=[col for col in processed.columns if 'unknown' in col])
    return cast_data_types(processed)

def process_unlabeled(df, column_mapping):
    processed = df.copy()
    processed.columns = [column_mapping.get(i, 'Unknown') for i in range(len(df.columns))]
    processed = processed.drop(columns=[col for col in processed.columns if 'unknown' in col])
    return cast_data_types(processed)

def cast_data_types(df):
    # Date handling
    date_cols = [col for col in df.columns if 'Date' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Numeric handling
    num_cols = [col for col in df.columns if col in ['Credit', 'Debit', 'Balance']]
    for col in num_cols:
        df[col] = df[col].replace(r'[^\d.-]', '', regex=True)
        df[col] = df[col].replace(r'^\.|\.$', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Text handling
    text_cols = [col for col in df.columns if 'Description' in col]
    for col in text_cols:
        df[col] = df[col].astype(str)
    
    return df

def default_processing(dataframes):
    # Fallback processing when no labels found
    processed = []
    for df in dataframes:
        # Simple column type inference
        type_scores = []
        for col in df.columns:
            col_data = df[col].dropna()
            scores = {
                'Date': pd.to_datetime(col_data, errors='coerce').notna().mean(),
                'Numeric': pd.to_numeric(col_data, errors='coerce').notna().mean(),
                'Text': (col_data.astype(str).str.len().mean() > 10) * 1.0
            }
            type_scores.append(scores)
        
        # Assign best matching types
        column_types = []
        for scores in type_scores:
            if scores['Date'] > 0.8:
                column_types.append('Date')
            elif scores['Numeric'] > 0.7:
                column_types.append('Numeric')
            else:
                column_types.append('Description')
        
        # Create standard dataframe
        std_df = pd.DataFrame()
        date_cols = [col for col, t in zip(df.columns, column_types) if t == 'Date']
        std_df['Date'] = df[date_cols[0]] if date_cols else pd.NaT
        
        desc_cols = [col for col, t in zip(df.columns, column_types) if t == 'Description']
        std_df['Description'] = df[desc_cols[0]] if desc_cols else 'Unknown'
        
        num_cols = [col for col, t in zip(df.columns, column_types) if t == 'Numeric']
        for i, col in enumerate(num_cols[:3]):
            std_df[['Credit', 'Debit', 'Balance'][i]] = df[col]
        
        processed.append(cast_data_types(std_df))
    
    return processed

if __name__ == '__main__':
    with open('bank_statements/statement_3.pkl', 'rb') as f:
        dataframes = pkl.load(f)
    processed_dfs = process_bank_statements(dataframes)
    print(processed_dfs)