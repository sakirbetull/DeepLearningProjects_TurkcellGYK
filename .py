'''
Cost-sensitive Learning – Maliyet Duyarlı Öğrenme
Amaç: Tüm hataların maliyeti eşit değildir. Özellikle iade edilen ürünler firmaya daha pahalıya mal olur.

Yaklaşım: Modelde yanlış tahminlerin maliyetine göre bir ağırlıklı kayıp fonksiyonu tanımlanabilir.

💰 Örnek:
Bir model yanlış şekilde "ürün iade edilmez" derse ve iade edilirse, maliyet 200 TL olabilir.
Bu durumda "False Negative" hatası daha pahalıdır ve buna göre model eğitilmelidir.

Explainable AI (XAI) – Açıklanabilir Yapay Zeka
Amaç: Modelin neden belirli bir karar verdiğini anlamak ve açıklamak.

Yöntemler:

SHAP: Özelliklerin karara katkısını gösterir.

LIME: Lokal olarak açıklama yapar (tek bir tahmin için).

🔍 Örnek:
Model “Bu müşteri dolandırıcı olabilir” dedi. SHAP gösteriyor ki:

Adres daha önce dolandırıcılıkla eşleşmiş → +0.45

Sipariş tutarı çok yüksek → +0.30

Yeni kullanıcı → +0.20
Sonuç: Yüksek risk.
'''


import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Veritabanı bağlantısı ve SQL sorgusu
def load_data():
    user = 'postgres'
    password = '12345'
    host = 'localhost'
    port = '5432'
    database = 'gyk'

    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')
    query = """
    WITH customer_stats AS (
        SELECT 
            c.customer_id,
            c.company_name,
            COUNT(DISTINCT o.order_id) as total_orders,
            SUM(od.quantity * od.unit_price * (1 - od.discount)) as total_spent,
            MAX(o.order_date) as last_order_date,
            AVG(od.quantity * od.unit_price * (1 - od.discount)) as avg_order_value,
            CASE 
                WHEN MAX(o.order_date) >= (SELECT MAX(order_date) FROM orders) - INTERVAL '6 months' 
                THEN 1 
                ELSE 0 
            END as will_order_again
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN order_details od ON o.order_id = od.order_id
        GROUP BY c.customer_id, c.company_name
    )
    SELECT 
        customer_id,
        company_name,
        total_orders,
        total_spent,
        last_order_date,
        avg_order_value,
        will_order_again,
        EXTRACT(MONTH FROM last_order_date) as order_month,
        CASE 
            WHEN EXTRACT(MONTH FROM last_order_date) IN (6, 7, 8, 12) 
            THEN 1 
            ELSE 0 
        END as is_high_season,
        CASE 
            WHEN total_spent < (SELECT PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY total_spent) FROM customer_stats) THEN 'Bronze'
            WHEN total_spent < (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY total_spent) FROM customer_stats) THEN 'Silver'
            WHEN total_spent < (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY total_spent) FROM customer_stats) THEN 'Gold'
            ELSE 'Platinum'
        END as customer_segment
    FROM customer_stats;
    """
    df = pd.read_sql_query(query, engine)
    return df
    print(df.head())
    print(df.info())
    print(df.describe().T)

# Boş değerleri doldurma
def preprocess_data(df):
    df['total_spent'].fillna(0, inplace=True)
    df['total_orders'].fillna(0, inplace=True)
    df['avg_order_value'].fillna(0, inplace=True)
    df['last_order_date'].fillna(pd.Timestamp("1970-01-01"), inplace=True)
    df['order_month'] = pd.to_datetime(df['last_order_date']).dt.month

    # Mevsim sütunu oluştur
    def map_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
    df['season'] = df['order_month'].apply(map_season)


    # Dağılım grafikleri
    columns_to_plot = ['total_orders', 'total_spent', 'avg_order_value', 'order_month']
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(columns_to_plot, 1):
        plt.subplot(1, len(columns_to_plot), i)
        sns.histplot(df[column], kde=True, bins=20, color='blue')
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Log dönüşümü ve standartlaştırma
    df['total_spent_log'] = np.log1p(df['total_spent'])
    df['avg_order_value_log'] = np.log1p(df['avg_order_value'])

    scaler = StandardScaler()
    df['total_orders_scaled'] = scaler.fit_transform(df[['total_orders']])
    df['total_spent_scaled'] = scaler.fit_transform(df[['total_spent_log']])
    df['avg_order_value_scaled'] = scaler.fit_transform(df[['avg_order_value_log']])

    # Gereksiz sütunları kaldır
    df = df.drop(columns=['total_orders', 'total_spent', 'total_spent_log', 'avg_order_value', 'avg_order_value_log', 'last_order_date', 'customer_id', 'company_name', 'order_month'])
    return df

#   print(df['customer_segment'].value_counts())
#   print(df['will_order_again'].value_counts())---dengesizlik var

# Encoding işlemleri
def encode_features(df):
    # customer_segment için Label Encoding
    le = LabelEncoder()
    df['customer_segment'] = le.fit_transform(df['customer_segment'])

    # season için One-Hot Encoding
    df = pd.get_dummies(df, columns=['season'], drop_first=True)
    return df

# Derin öğrenme modeli
def build_and_train_model(X_train, y_train, X_test, y_test, class_weights):
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=16, class_weight=class_weights, validation_data=(X_test, y_test))
    return model

def main():
    # Veriyi yükle
    df = load_data()
    print("Veri seti yüklendi.")
    print(df.head())

    # Veriyi ön işleme
    df = preprocess_data(df)
    print("Veri ön işlendi.")
    print(df.info())

    # Encoding işlemleri
    df = encode_features(df)
    print("Encoding işlemleri tamamlandı.")
    print(df.head())

    # Özellikler ve hedef değişken
    X = df.drop(columns=['will_order_again'])
    y = df['will_order_again']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Class weight hesaplama
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # Model oluştur ve eğit
    model = build_and_train_model(X_train, y_train, X_test, y_test, class_weights)

if __name__ == "__main__":
    main()




    '''
     #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   will_order_again        91 non-null     int64
 1   order_month             91 non-null     int32
 2   is_high_season          91 non-null     int64
 3   customer_segment        91 non-null     object
 4   total_orders_scaled     91 non-null     float64
 5   total_spent_scaled      91 non-null     float64
 6   avg_order_value_scaled  91 non-null     float64

 will_order_again 
1    84
0     7

customer_segment 
Platinum    25
Silver      22
Bronze      22
Gold        22

    '''
