import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Union, List
import time
import re
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

class StatementInsights:
    def __init__(self):
        """
        Initialize the StatementInsights class with configurations
        """        
        # Refined financial taxonomy for classification
        self.categories = [
            'stable income', 'volatile income', 'debt payment',
            'essential expense', 'discretionary expense',
            'savings', 'uncategorized'
        ]
        
        # Comprehensive mapping of API categories to our defined categories
        self.category_mapping = {
            # Income categories
            'primary paycheck': 'stable income',
            'business income': 'volatile income',
            'repayment from others': 'volatile income',
            'other income': 'volatile income',
            
            # Essential Expenses
            'bills & utilities': 'essential expense',
            'auto & transport': 'essential expense',
            'vehicle & repairs': 'essential expense',
            'gas': 'essential expense',
            'food & drink': 'essential expense',
            'groceries': 'essential expense',
            'health & wellness': 'essential expense',
            'medical': 'essential expense',
            'other transportation': 'essential expense',
            
            # Discretionary Expenses
            'gym': 'discretionary expense',
            'restaurants & other': 'discretionary expense',
            'travel & vacation': 'discretionary expense',
            'shopping': 'discretionary expense',
            'clothing': 'discretionary expense',
            'entertainment & lifestyle': 'discretionary expense',
            'gifts & donations': 'discretionary expense',
            'subscriptions': 'discretionary expense',
            'other shopping': 'discretionary expense',
            'other health & wellness': 'discretionary expense',
            'education': 'discretionary expense',
            
            # Debt and Financial
            'loans & financial fees': 'debt payment',
            'credit card payment': 'debt payment',
            'taxes': 'debt payment',
            'insurance': 'debt payment',
            
            # Savings
            'investments': 'savings',
            # Transfers and Misc
            'transfer': 'savings',
            
            # Family and Personal
            'family & pets': 'discretionary expense',
            
            'other expenses': 'uncategorized'
        }

        # Load API credentials from environment variables
        self.API_KEY = 'fina-api-test'
        self.PARTNER_ID = 'f-f7rh58s2'
        
        if not self.API_KEY or not self.PARTNER_ID:
            print("Warning: Fina API credentials not set. Categorization will be limited.")

    def map_category(self, api_category: str) -> str:

        # Use mapping, default to 'uncategorized' if not found
        return self.category_mapping.get(api_category.lower(), 'uncategorized')

    def categorize_transactions_batch(self, transactions: List[str]) -> List[str]:

        # If no transactions, return empty list
        if not transactions:
            return []
        
        # Chunk size slightly below API limit for safety
        CHUNK_SIZE = 90
        
        # Prepare to collect all categories
        all_categories = []
        
        # Check API credentials
        if not self.API_KEY or not self.PARTNER_ID:
            print("API credentials not set. Defaulting to 'uncategorized'.")
            return ['uncategorized'] * len(transactions)
        
        # Split transactions into chunks
        for i in range(0, len(transactions), CHUNK_SIZE):
            chunk = transactions[i:i+CHUNK_SIZE]
            
            try:
                start_time = time.time()
                
                url = "https://app.fina.money/api/resource/categorize"
            
                headers = {
                "Content-Type": "application/json",
                "x-api-key": self.API_KEY,
                "x-partner-id": self.PARTNER_ID
                }
                
                # Send batch of transactions to API
                response = requests.post(url, json=chunk, headers=headers)
                response.raise_for_status()
                
                # Extract categories from response
                chunk_categories = response.json()
                
                # If API returns fewer categories than chunk, pad with 'uncategorized'
                if len(chunk_categories) < len(chunk):
                    chunk_categories.extend(['uncategorized'] * (len(chunk) - len(chunk_categories)))
                
                # Extend all categories list
                all_categories.extend(chunk_categories[:len(chunk)])
                
                # Optional: Log time taken for each chunk
                print(f"Categorized chunk of {len(chunk)} transactions in {time.time() - start_time:.2f} seconds")
            
            except requests.RequestException as e:
                print(f"Fina API batch categorization failed for chunk: {e}")
                
                # Fallback to uncategorized for this chunk
                all_categories.extend(['uncategorized'] * len(chunk))
        
        # Ensure we return exactly the same number of categories as input transactions
        assert len(all_categories) == len(transactions), "Mismatch in number of categories"
        
        return all_categories

    def _correct_ocr_dates(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Correct OCR misread dates by identifying and correcting significant date outliers
        
        Args:
            transactions: DataFrame with Date column
        
        Returns:
            DataFrame with corrected dates
        """
        # Ensure Date is datetime
        
        transactions['Date'] = pd.to_datetime(transactions['Date'], errors='coerce')
        
        # Remove any NaT values
        transactions = transactions.dropna(subset=['Date'])
        
        # If no transactions, return empty DataFrame
        if len(transactions) == 0:
            return transactions
        
        # Calculate date statistics
        date_series = transactions['Date']
        median_date = date_series.median()
        
        # Calculate Interquartile Range (IQR) for dates converted to numeric
        Q1 = date_series.quantile(0.25)
        Q3 = date_series.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (1.5 * IQR method)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outlier dates
        outliers = transactions[
            (transactions['Date'] < lower_bound) | 
            (transactions['Date'] > upper_bound)
        ]
        
        # Correction strategy
        if len(outliers) > 0:
            print("\n--- Date Outlier Correction ---")
            print(f"Detected {len(outliers)} date outliers")
            print("Outlier dates:")
            for idx, row in outliers.iterrows():
                original_date = row['Date']
                
                # Attempt to correct by adjusting the year
                # Find the closest reasonable year based on the median date
                year_diff = median_date.year - original_date.year
                corrected_date = original_date.replace(year=original_date.year + year_diff)
                
                # Update the date in the DataFrame
                transactions.at[idx, 'Date'] = corrected_date
                
                print(f"Corrected: {original_date} → {corrected_date}")
        
        # Re-sort by date after corrections
        transactions = transactions.sort_values('Date')
        
        print("\nFinal date range after correction: "
              f"{transactions['Date'].min()} to {transactions['Date'].max()}")
        
        return transactions

    def preprocess_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Preprocessing Transactions ---")
        print(f"Initial DataFrame shape: {transactions.shape}")
        print(f"Initial columns: {list(transactions.columns)}")
        
        # Ensure Date column is in datetime format
        transactions['Date'] = pd.to_datetime(transactions['Date'], infer_datetime_format=True, errors='coerce')
        print(f"Rows after datetime conversion: {len(transactions)}")
        
        # Drop rows with invalid dates
        transactions = transactions.dropna(subset=['Date'])
        print(f"Rows after dropping invalid dates: {len(transactions)}")
        
        # Correct OCR misread dates
        transactions = self._correct_ocr_dates(transactions)
        
        # Sort transactions by Date
        transactions = transactions.sort_values('Date')
        print(f"Date range: {transactions['Date'].min()} to {transactions['Date'].max()}")
        
        # Clean descriptions
        transactions['Description'] = transactions['Description'].apply(self._clean_description)
        print(f"Rows after description cleaning: {len(transactions)}")

        # Batch categorize transactions
        start_time = time.time()
        categories = self.categorize_transactions_batch(transactions['Description'].tolist())
        
        # Add categories to DataFrame
        transactions['Category'] = categories
        
        categorization_time = time.time() - start_time
        print(f"Total categorization time: {categorization_time:.2f} seconds")
        
        print(f"Category distribution:\n{transactions['Category'].value_counts()}")

        # display each transaction with its category
        for i, row in transactions.iterrows():
            print(f"Transaction: {row['Description']}, Category: {row['Category']}")

        print(f"\nFinal DataFrame shape: {transactions.shape}")
        return transactions

    def _clean_description(self, description: str) -> str:
        """Clean transaction description to focus on merchant name."""
        # Convert to uppercase for consistency
        desc = description.upper()
        
        # Remove common transaction codes and card numbers (patterns like CD XXXX)
        desc = re.sub(r'\bCD\s+\d+\b', '', desc)
        
        # Remove common date patterns
        desc = re.sub(r'\d{2}[A-Z]{3}\d{2}', '', desc)
        
        # Remove reference numbers and other common transaction codes
        desc = re.sub(r'RP\d+', '', desc)
        
        # Remove multiple spaces and trim
        desc = ' '.join(desc.split())
        
        return desc

    def get_frequent_transactions(self, min_frequency: int = 3, similarity_threshold: float = 0.96, k: int = 5) -> List[Dict[str, Any]]:
        if len(self.transactions) == 0:
            return []
            
        descriptions = self.transactions['Description'].fillna('').tolist()
        # Convert descriptions to TF-IDF vectors with custom parameters
        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words='english',
            min_df=2,  # Ignore terms that appear in less than 2 documents
            ngram_range=(1, 2),  # Use both unigrams and bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Convert similarity to distance (ensure non-negative)
        distance_matrix = np.maximum(0, 1 - similarity_matrix)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=1 - similarity_threshold,  # Convert similarity threshold to distance
            min_samples=min_frequency,
            metric='precomputed'
        )
        
        clusters = clustering.fit_predict(distance_matrix)
        
        # Process clusters to get frequent transaction groups
        frequent_groups = []
        for cluster_id in set(clusters):
            if frequent_groups and len(frequent_groups) >= k:
                break
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get indices of transactions in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Get transaction details
            cluster_transactions = self.transactions.iloc[cluster_indices]
            
            # Calculate total amount (Credit - Debit)
            total_amount = cluster_transactions['Credit'].fillna(0).sum() - cluster_transactions['Debit'].fillna(0).sum()
            
            # Create summary for this group
            group_summary = {
                'transactions': cluster_transactions['Description'].tolist(),
                'frequency': len(cluster_indices),
                'total_amount': total_amount,
                'average_amount': total_amount / len(cluster_indices),
                'category': cluster_transactions['Category'].mode().iloc[0]
            }
            
            frequent_groups.append(group_summary)
        
        # Sort groups by frequency (descending)
        frequent_groups.sort(key=lambda x: x['frequency'], reverse=True)
        
        return frequent_groups

    def analyze_transactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Calculate totals
        total_debits = df['Debit'].sum()
        total_credits = df['Credit'].sum()
        
        # Category analysis
        category_spending = df.groupby('Category')['Debit'].sum().sort_values(ascending=False)
        category_income = df.groupby('Category')['Credit'].sum().sort_values(ascending=False)
        
        # Daily analysis
        daily_net = df.groupby('Date').agg({
            'Credit': 'sum',
            'Debit': 'sum',
            'Balance': 'last'
        })
        daily_net['Net'] = daily_net['Credit'] - daily_net['Debit']
        
        # Transaction statistics
        avg_debit = df[df['Debit'] > 0]['Debit'].mean()
        avg_credit = df[df['Credit'] > 0]['Credit'].mean()
        largest_expense = df[df['Debit'] > 0]['Debit'].max()
        largest_income = df[df['Credit'] > 0]['Credit'].max()
        
        # Balance analysis
        balance_change = df['Balance'].iloc[-1] - df['Balance'].iloc[0]
        balance_volatility = df['Balance'].std()
        
        return {
            'total_debits': total_debits,
            'total_credits': total_credits,
            'net_change': total_credits - total_debits,
            'category_spending': category_spending,
            'category_income': category_income,
            'daily_summary': daily_net,
            'avg_transaction_size': {
                'debit': avg_debit,
                'credit': avg_credit
            },
            'largest_transactions': {
                'expense': largest_expense,
                'income': largest_income
            },
            'balance_metrics': {
                'start_balance': df['Balance'].iloc[0],
                'end_balance': df['Balance'].iloc[-1],
                'change': balance_change,
                'volatility': balance_volatility
            }
        }
        
    def plot_balance_trends(self, df: pd.DataFrame) -> None:
        """
        Create visualizations for monthly statement analysis
        
        Args:
            df: DataFrame with columns: Date, Description, Credit, Debit, Balance, Category
        """
        # Create a figure with 4 subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Balance Over Time
        ax1 = plt.subplot(221)
        ax1.plot(df['Date'], df['Balance'], linewidth=2, marker='o')
        ax1.set_title('Daily Balance')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Balance')
        
        # 2. Category Spending Distribution
        ax2 = plt.subplot(222)
        category_spending = df.groupby('Category')['Debit'].sum().sort_values(ascending=True)
        category_spending.plot(kind='barh', ax=ax2)
        ax2.set_title('Spending by Category')
        ax2.set_xlabel('Total Spent')
        
        # 3. Daily Credits vs Debits
        ax3 = plt.subplot(223)
        daily_summary = df.groupby('Date').agg({
            'Credit': 'sum',
            'Debit': 'sum'
        })
        daily_summary.plot(kind='bar', ax=ax3)
        ax3.set_title('Daily Credits vs Debits')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Amount')
        
        # Truncate x-axis labels for Daily Credits vs Debits
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit number of x-ticks
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: daily_summary.index[int(x)].strftime('%d-%b') if 0 <= int(x) < len(daily_summary) else ''
        ))
        
        # 4. Category Transaction Counts
        ax4 = plt.subplot(224)
        df['Transaction_Count'] = 1
        category_counts = df.groupby('Category')['Transaction_Count'].count().sort_values(ascending=True)
        category_counts.plot(kind='barh', ax=ax4)
        ax4.set_title('Transaction Count by Category')
        ax4.set_xlabel('Number of Transactions')
        
        plt.tight_layout()
        plt.show()

    def get_balance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        df['Transaction_Count'] = 1
        balance_trends = {
            'balance_over_time': [
                {
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Balance': float(row['Balance'])
                }
                for _, row in df[['Date', 'Balance']].iterrows()
            ],
            'category_spending': {
                k: float(v) 
                for k, v in df.groupby('Category')['Debit'].sum().sort_values(ascending=True).to_dict().items()
            },
            'daily_credits_vs_debits': [
                {
                    'Date': date.strftime('%Y-%m-%d'),
                    'Credit': float(data['Credit']),
                    'Debit': float(data['Debit'])
                }
                for date, data in df.groupby('Date').agg({
                    'Credit': 'sum',
                    'Debit': 'sum'
                }).iterrows()
            ],
            'category_transaction_counts': {
                k: int(v)
                for k, v in df.groupby('Category')['Transaction_Count'].count().sort_values(ascending=True).to_dict().items()
            }
        }
        return balance_trends

    def calculate_loan_fitness(self, transactions_df):
        """Calculate a loan fitness score based on financial metrics."""
        scores = {
            'balance_stability': 0,
            'income_consistency': 0,
            'spending_discipline': 0,
            'savings_ratio': 0
        }
        
        # 1. Balance Stability (25 points)
        avg_balance = transactions_df['Balance'].mean()
        balance_std = transactions_df['Balance'].std()
        if avg_balance > 0:
            cv = (balance_std / avg_balance) if avg_balance != 0 else float('inf')
            scores['balance_stability'] = min(25, 25 * (1 / (1 + cv)))
        
        # 2. Income Consistency (25 points)
        credits = transactions_df[transactions_df['Credit'].notna() & (transactions_df['Credit'] > 0)]['Credit']
        if not credits.empty:
            income_cv = credits.std() / credits.mean() if credits.mean() != 0 else float('inf')
            scores['income_consistency'] = min(25, 25 * (1 / (1 + income_cv)))
        
        # 3. Spending Discipline (25 points)
        debits = transactions_df[transactions_df['Debit'].notna()]['Debit']
        if not debits.empty:
            debits_abs = abs(debits)
            avg_spending = debits_abs.mean()
            spending_cv = debits_abs.std() / avg_spending if avg_spending != 0 else float('inf')
            scores['spending_discipline'] = min(25, 25 * (1 / (1 + spending_cv)))
        
        # 4. Savings Ratio (25 points)
        total_credits = credits.sum() if not credits.empty else 0
        total_debits = abs(debits).sum() if not debits.empty else 0
        if total_credits > 0:
            savings_ratio = (total_credits - total_debits) / total_credits
            scores['savings_ratio'] = min(25, max(0, 25 * (savings_ratio + 0.1)))
        
        # Calculate total score and risk level
        total_score = sum(scores.values())
        
        risk_assessment = {
            'score': round(total_score, 2),
            'max_score': 100,
            'risk_level': 'High' if total_score < 50 else 'Medium' if total_score < 75 else 'Low',
            'component_scores': {k: round(v, 2) for k, v in scores.items()},
            'interpretation': {
                'balance_stability': 'Measures how stable your account balance remains over time',
                'income_consistency': 'Evaluates the regularity and predictability of income',
                'spending_discipline': 'Assesses spending patterns and financial discipline',
                'savings_ratio': 'Measures ability to save and maintain positive cash flow'
            }
        }
        
        return risk_assessment

    def execute(self, transactions: Union[pd.DataFrame, List[pd.DataFrame]]) -> Dict[str, Any]:
        print("\n=== Executing Statement Insights ===")
        # Validate input
        if not transactions:
            raise ValueError("No transactions provided")
        
        # If transactions is a list, concatenate all DataFrames
        if isinstance(transactions, list):
            print(f"Received {len(transactions)} DataFrames")
            transactions = pd.concat(transactions, ignore_index=True)
        
        # Validate required columns
        required_columns = ['Date', 'Description', 'Credit', 'Debit', 'Balance']
        missing_columns = [col for col in required_columns if col not in transactions.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.transactions = transactions

        # Preprocess transactions
        self.transactions = self.preprocess_transactions(self.transactions)

        # Get frequent transactions
        frequent_transactions = self.get_frequent_transactions()
        
        # Analyze transactions
        analysis_results = self.analyze_transactions(self.transactions)
        
        # Generate balance trend data
        balance_trends = self.get_balance_trends(self.transactions)
        
        # Calculate loan fitness score
        loan_fitness = self.calculate_loan_fitness(self.transactions)
        
        # Structure the insights in a display-friendly format
        structured_insights = {
            "summary": {
                "total_transactions": len(self.transactions),
                "date_range": {
                    "start": self.transactions['Date'].min().strftime('%Y-%m-%d'),
                    "end": self.transactions['Date'].max().strftime('%Y-%m-%d')
                },
                "total_credits": float(analysis_results.get('total_credits', 0)),
                "total_debits": float(analysis_results.get('total_debits', 0)),
                "net_change": float(analysis_results.get('net_change', 0))
            },
            "frequent_transactions": [
                {
                    "description": group['transactions'][0],
                    "frequency": int(group['frequency']),
                    "average_amount": float(group['average_amount']),
                    "total_amount": float(group['total_amount'])
                }
                for group in frequent_transactions[:5]  # Top 5 frequent transactions
            ],
            "spending_insights": {
                "average_transaction": float(analysis_results.get('avg_transaction_size', {}).get('debit', 0)),
                "largest_credit": float(analysis_results.get('largest_transactions', {}).get('income', 0)),
                "largest_debit": float(analysis_results.get('largest_transactions', {}).get('expense', 0)),
                "average_deposit": float(analysis_results.get('avg_transaction_size', {}).get('credit', 0))
            },
            "balance_trends": balance_trends,
            "loan_fitness": loan_fitness
        }
        
        return structured_insights


if __name__ == "__main__":
    transactions = pd.DataFrame({
        'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
        'Description': ['Salary', 'Rent', 'Groceries'],
        'Credit': [1000, 0, 0],
        'Debit': [0, 500, 200],
        'Balance': [1000, 500, 300],
    })

    # Initialize the StatementInsights class
    statement_analyzer = StatementInsights()
    
    # Execute insights generation
    statement_analyzer.execute(transactions)