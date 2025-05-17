# Northwind Veritabanı Derin Öğrenme Projeleri

Bu repository, Northwind veritabanı üzerinde gerçekleştirilen üç farklı derin öğrenme projesini içermektedir. Her proje, iş dünyasındaki farklı problemlere derin öğrenme çözümleri sunmaktadır.

## Projeler

### 1. Sipariş Verme Alışkanlığı Tahmini
Müşterilerin gelecekteki sipariş davranışlarını tahmin eden bir derin öğrenme modeli.

**Özellikler:**
- Toplam harcama analizi
- Sipariş sayısı ve ortalama sipariş büyüklüğü değerlendirmesi
- 6 aylık tahmin periyodu
- Mevsimsellik etkisi analizi
- Veri artırma teknikleri
- Sınıf dengesizliği çözümleri

### 2. Ürün İade Risk Skoru
Siparişlerin iade edilme riskini tahmin eden yapay zeka modeli.

**Özellikler:**
- İndirim oranı analizi
- Ürün miktarı ve harcama değerlendirmesi
- Maliyet duyarlı öğrenme
- Açıklanabilir Yapay Zeka (XAI) entegrasyonu
- SHAP ve LIME ile model açıklamaları

### 3. Yeni Ürün Satın Alma Potansiyeli
Müşterilerin yeni ürünlere olan ilgisini tahmin eden sinir ağı modeli.

**Özellikler:**
- Kategori bazlı satın alma analizi
- Derin öğrenme tabanlı öneri sistemi
- Çoklu etiket tahmini
- Neural Collaborative Filtering
- AutoEncoder tabanlı öneriler

## Teknik Detaylar

### Veri Kaynağı
- Northwind Veritabanı
- Kullanılan tablolar: Orders, Order Details, Customers, Products, Categories

### Gereksinimler
- Python 3.8+
- TensorFlow 2.x
- PyTorch
- scikit-learn
- pandas
- numpy
- SHAP
- LIME



## Proje Yapısı
```
her projenin içinde kendine özgü proje yapısı bulunmaktadır.
```

## Kullanım
Her proje kendi dizininde bağımsız olarak çalıştırılabilir. Detaylı kullanım talimatları ilgili proje dizinlerindeki README dosyalarında bulunmaktadır.

## Katkıda Bulunma
1. Bu repository'yi fork edin
2. Feature branch'i oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans
Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## İletişim
Betül Şakır - sakirbetul@outlook.com

Proje Linki: https://github.com/sakirbetull/DeepLearningProjects_TurkcellGYK.git





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

2. Örnek Soru
Ürün İade Risk Skoru

Müşterilerin daha önceki siparişlerindeki indirim oranı, ürün miktarı ve harcama miktarına göre bir siparişin iade edilme riskini tahmin eden bir derin öğrenme modeli oluştur.

İpucu:
Order Details tablosunda Discount bilgisi var.

(Northwind küçük olduğu için) İade olayını yüksek indirim + düşük harcama gibi bir mantıkla sahte etiketleyebilirsin.

Ar-Ge Konuları:
Cost-sensitive Learning: İade edilen ürünlerin firmaya maliyeti daha yüksek. Modeli bu durumu daha ciddiye alacak şekilde ağırlıklandır.

Explainable AI (XAI): SHAP veya LIME gibi yöntemlerle "Model neden bu siparişi riskli buldu?" açıklamasını çıkar.

3. Örnek Soru
Yeni Ürün Satın Alma Potansiyeli

Müşterilerin geçmiş satın alma kategorilerine (örneğin "Beverages", "Confections") bakarak, yeni çıkan bir ürünü satın alma ihtimallerini tahmin eden bir sinir ağı modeli geliştir.

İpucu:
Products, Categories, Order Details ve Orders tablolarını birleştir.

Müşterinin hangi kategorilerde ne kadar harcama yaptığı gibi özellikler üret.

Ar-Ge Konuları:
Recommendation Systems: Deep Learning tabanlı ürün öneri sistemleri araştır (örneğin Neural Collaborative Filtering, AutoEncoders).

Multi-label Prediction: Aynı anda birkaç ürünü birden önerebilecek bir sistem geliştir.

****Tüm örnekleri api haline getiriniz.

'''