# database.py

import psycopg2
import pandas as pd
from config import DB_CONFIG

class DatabaseHandler:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print("Successfully connected to database")
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            raise

    def get_historical_data(self, symbol, start_date, end_date):
        """Get historical data with precise hourly timestamps"""
        query = """
            SELECT 
                date_time,
                open_price,
                high_price,
                low_price,
                close_price,
                volume_crypto,
                volume_usd
            FROM crypto_data_hourly
            WHERE symbol = %s
              AND date_time >= %s::timestamp
              AND date_time <= %s::timestamp
            ORDER BY date_time ASC
        """
        
        try:
            print(f"\nFetching data for {symbol}:")
            print(f"Start: {start_date}")
            print(f"End: {end_date}")
            
            df = pd.read_sql_query(
                query,
                self.conn,
                params=(symbol, start_date, end_date),
                parse_dates=['date_time']
            )
            
            # Set date_time as the DataFrame index
            df.set_index('date_time', inplace=True)
            
            print(f"Fetched {len(df)} hourly records")
            if len(df) > 0:
                print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
                
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed.")
