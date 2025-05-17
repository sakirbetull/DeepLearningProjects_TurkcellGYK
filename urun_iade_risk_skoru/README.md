# Ürün İade Risk Skoru Projesi

Bu proje, e-ticaret sistemlerinde ürün iade riskini tahmin etmek için geliştirilmiş bir makine öğrenmesi modelidir.

## Proje Hakkında

Proje, müşteri sipariş verilerini analiz ederek, hangi ürünlerin iade edilme riskinin yüksek olduğunu tahmin eder. Model, aşağıdaki faktörleri dikkate alır:

- Sipariş tutarı
- İndirim oranları
- Müşteri alışveriş geçmişi
- Ürün kategorisi
- Sipariş miktarı

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. `.env` dosyasını oluşturun ve veritabanı bağlantı bilgilerinizi ekleyin:
```
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

## Proje Yapısı

```
urun_iade_risk_skoru/
├── src/
│   ├── model.py           # Model tanımlamaları ve eğitim fonksiyonları
│   ├── feature_engineering.py  # Özellik mühendisliği işlemleri
│   └── database.py        # Veritabanı işlemleri
├── config.py              # Konfigürasyon ayarları
├── main.py               # Ana uygulama dosyası
└── README.md             # Bu dosya
```

## Kullanım

Projeyi çalıştırmak için:

```bash
python main.py
```

## Model Detayları

Model, aşağıdaki özellikleri kullanarak tahmin yapar:

- Birim fiyat
- Miktar
- İndirim oranı
- Toplam tutar
- İndirimli tutar
- Ortalama sipariş tutarı
- Sipariş tutarı standart sapması
- Toplam harcama
- Ortalama indirim
- Maksimum indirim
- Ortalama miktar
- Toplam miktar

## Model Mimarisi

Model, TensorFlow/Keras kullanılarak geliştirilmiş bir yapay sinir ağıdır:

- Giriş katmanı
- 64 nöronlu ReLU aktivasyonlu gizli katman
- 32 nöronlu ReLU aktivasyonlu gizli katman
- 16 nöronlu ReLU aktivasyonlu gizli katman
- Sigmoid aktivasyonlu çıkış katmanı

## Geliştirici Notları

- Model eğitimi sırasında erken durdurma (early stopping) kullanılmaktadır
- En iyi model ağırlıkları otomatik olarak kaydedilmektedir
- Veri ön işleme için StandardScaler kullanılmaktadır

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 