import pandas as pd
import psycopg2
from urun_iade_risk_skoru.config import DB_CONFIG

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
    # defence coding hata sonunda hatayı yazması
    def connect(self):
        try:
            self.connection = psycopg2.connect(**DB_CONFIG)
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise
    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def get_order_data(self):
        query = """
            SELECT 
                od.order_id, 
                od.product_id, 
                od.discount, 
                od.quantity, 
                od.unit_price 
                o.customer_id,
                o.order_date,
                p.category_id,
                c.company_name

            FROM orders o
            INNER JOIN order_details od 
            ON o.order_id = od.order_id
            INNER JOIN products p
            ON od.product_id = p.product_id
            INNER JOIN customers c
            ON o.customer_id = c.customer_id
        """
        try:
            df = pd.read_sql_query(query, self.connection)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            raise


