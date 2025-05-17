'''
1. Örnek Soru
Sipariş Verme Alışkanlığı Tahmini

Northwind veritabanında müşterilerin toplam harcaması, sipariş sayısı ve ortalama sipariş büyüklüğüne göre bir müşterinin önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin eden bir derin öğrenme modeli kur.

İpucu:
Veritabanından Orders, Order Details, Customers tablolarını kullan.

"Son sipariş tarihi" bilgisine göre 6 ay sınırı belirle.

Ar-Ge Konuları:
Temporal Features: Mevsimsellik etkisi var mı? (Örn: Yaz aylarında sipariş artıyor mu?)

Data Augmentation: Müşteri datasını arttırarak daha büyük bir veri seti oluşturup modelin başarısını gözlemle.

Class Imbalance: Eğer az kişi sipariş veriyorsa, class_weight veya SMOTE gibi yöntemlerle çözüm üret.


'''

import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# cpnnect to the PostgreSQL database
connection = psycopg2.connect(
    host="localhost",
    database="gyk",
    user="postgres",
    password="12345", port=5432
)
query = """
with last_order_date as (
    select max(order_date) as max_date from orders
), 
customer_stats as (
    select 
        c.customer_id,
        count(o.order_id) as total_orders,
        sum(od.unit_price * od.quantity) as total_spent,
        avg(od.unit_price * od.quantity) as avg_order_value
    from orders o
    inner join customers c on o.customer_id = c.customer_id
    inner join order_details od on o.order_id = od.order_id
    group by c.customer_id
),
label as ( 
    select 
        c.customer_id,
        case 
            when exists (
                select 1 
                from orders o2, last_order_date lod
                where o2.customer_id = c.customer_id 
                and o2.order_date >= (lod.max_date - interval '6 months')
            ) 
            then 1 else 0 
        end as will_order_again
    from customers c
)
select 
    s.customer_id,
    s.total_orders,
    s.total_spent,
    s.avg_order_value,
    l.will_order_again
from customer_stats s 
join label l on s.customer_id = l.customer_id;
    """
df = pd.read_sql(query, connection)
connection.close()

# customer_id'yi kullanmadık çünkü modeldeki bir feature değil, sadece bir tanımlayıcı
X = df.drop(columns=['customer_id', 'will_order_again'])
y = df['will_order_again']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile belleğe alır
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")