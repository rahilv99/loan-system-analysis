from table_parser import TableParser
from standardize import process_bank_statements
import time
import pickle as pkl
import pandas as pd
from statement_insights import StatementInsights


if __name__ == "__main__":
    time_start = time.time()
    parser = TableParser()
    tables = parser.parse_pdf_tables('bank_statements/statement_1.pdf')
    
    time_end = time.time()
    print(f"Table parsing time: {time_end - time_start:.2f} seconds")
    
    print(f'Checkpoint 1: {tables}')

    time_start = time.time()
    results = process_bank_statements(tables)
    time_end = time.time()
    print(f"Table standardization time: {time_end - time_start:.2f} seconds")
    
    print(results)

    # Initialize the StatementInsights class
    statement_analyzer = StatementInsights()
    
    # Execute insights generation
    insights = statement_analyzer.execute(results[0])