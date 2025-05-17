# veritabanına erişim olmadığı durumlarda kullanılır. SQL ile veri çekmek daha sağlıklı bir yöntemdir.
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from urun_iade_risk_skoru.config import FEATURE_CONFIG

class FeatureEngineering:
    def __init__(self, df):
        self.scaler = StandardScaler()
        self.customer_features = None

    def create_features(self, df):
        df['total_amount'] = df['unit_price'] * df['quantity']* (1 - df['discount'])
        df['discounted_amount'] = df['unit_price'] * df['quantity'] * df['discount']

        self.customer_features = df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'std'],
            'discounted_amount': ['mean', 'max'],
            'quantity': ['sum', 'mean']}).reset_index()

        self.customer_features.columns = ['customer_id', 'avg_order_amount',
                                        'std_order_amount', 'total_spent',
                                        'avg_discount', 'max_discount',
                                        'avg_quantity', 'total_quantity']
        # empty columns
        df = df.merge(self.customer_features, on='customer_id', how='left')
        
        high_discount = df['discount'] > df['discount'].quantile(FEATURE_CONFIG['high_discount_threshold'])
        low_amount = df['total_amount'] < df['total_amount'].quantile(FEATURE_CONFIG['low_amount_threshold'])
        df['return_risk'] = (high_discount & low_amount).astype(int) 

        return df
    
    def prepare_model_data(self, df):
        feature_columns = ['unit_price', 'quantity', 'discount', 'total_amount', 'discounted_amount',
                            'avg_order_amount', 'std_order_amount', 'total_spent',
                            'avg_discount', 'max_discount', 'avg_quantity', 'total_quantity']
        
        X = df[feature_columns]
        y = df['return_risk']

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    