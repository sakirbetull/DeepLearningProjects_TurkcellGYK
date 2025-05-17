from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'gyk'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '12345'),  
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432)
}

MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'epochs': 100,
}

# comes from the domain knowledge of the problem
FEATURE_CONFIG = {
    'high_discount_threshold': 0.75, # 75th percentile for high discount begins
    'low_amount_threshold': 0.25
} 