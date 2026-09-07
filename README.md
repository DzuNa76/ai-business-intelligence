# AI E-Commerce Customer Analytics

End-to-end **AI & Data Science project** untuk mengolah data e-commerce menjadi business insights, customer segmentation, dan machine learning yang dapat mendukung pengambilan keputusan bisnis.

## Project Overview

Project ini mensimulasikan workflow seorang **AI Data Scientist / Data Scientist** di perusahaan e-commerce.

Fokus utama project:

* Exploratory Data Analysis (EDA)
* Business insight generation
* Customer behavior analysis
* RFM analysis
* Customer segmentation menggunakan Machine Learning
* Churn prediction
* Feature engineering
* Model evaluation
* Business-oriented interpretation

Project tidak hanya berfokus pada pembuatan model, tetapi pada bagaimana data diubah menjadi **informasi yang dapat digunakan untuk pengambilan keputusan bisnis**.

---

## Business Problem

Perusahaan e-commerce memiliki data mengenai:

* customer
* product
* transaction
* website/app session
* customer review

Namun data tersebut belum secara langsung menjawab pertanyaan bisnis seperti:

1. Bagaimana perkembangan revenue?
2. Produk atau kategori apa yang memberikan kontribusi revenue terbesar?
3. Seberapa terkonsentrasi revenue pada customer tertentu?
4. Apa perbedaan customer aktif dan churned?
5. Bagaimana hubungan engagement dengan conversion?
6. Negara mana yang memberikan performa terbaik?
7. Customer seperti apa yang memiliki nilai tinggi?
8. Customer mana yang berpotensi churn?
9. Bagaimana perusahaan dapat membuat strategi marketing yang lebih terarah?

Project ini mencoba menjawab pertanyaan tersebut menggunakan pendekatan **Data Analytics + Machine Learning**.

---

## Dataset

Dataset yang digunakan:

**Synthetic E-Commerce Behavior Dataset**

Dataset terdiri dari beberapa sumber data:

| Dataset        | Description                       |
| -------------- | --------------------------------- |
| `customers`    | Customer profile dan status churn |
| `products`     | Informasi produk dan kategori     |
| `transactions` | Riwayat transaksi                 |
| `sessions`     | Aktivitas customer pada platform  |
| `reviews`      | Customer review                   |

### Dataset Scale

Hasil data validation:

| Dataset                |                        Records |
| ---------------------- | -----------------------------: |
| Customers              |                         10,000 |
| Transactions           |                        120,000 |
| Sessions               |                         80,000 |
| Reviews                | Dataset tersedia pada raw data |
| Completed Transactions |                         68,700 |

### Data Quality

Beberapa validasi yang dilakukan:

* Duplicate customer ID: **0**
* Duplicate transaction ID: **0**
* Duplicate product ID: **0**
* Duplicate session ID: **0**
* Duplicate review ID: **0**
* Orphan foreign-key records: **0**
* Transaction status divalidasi
* Date range diperiksa
* Transaction amount divalidasi terhadap quantity, price, dan discount

---

# Project Workflow

Project mengikuti workflow Data Science end-to-end:

```text
Raw Data
   │
   ▼
Data Exploration
   │
   ▼
Data Preparation
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Business Insights
   │
   ▼
Feature Engineering
   │
   ├───────────────┐
   ▼               ▼
Customer       Churn
Segmentation   Prediction
   │               │
   ▼               ▼
Business       Model
Segments       Evaluation
   │               │
   └───────┬───────┘
           ▼
    Business Recommendations
```

---

# Project Structure

```text
ai-business-intelligence/
│
├── data/
│   ├── raw/
│   │   └── synthetic-ecommerce-behavior/
│   │
│   └── processed/
│       └── customer_features.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_customer_segmentation.ipynb
│   └── 06_churn_prediction.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── models/
│
├── reports/
│   ├── figures/
│   └── business_insights.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Struktur dapat berkembang selama proses pengembangan project.

---

# 1. Data Exploration

Tahap pertama digunakan untuk memahami struktur dataset sebelum melakukan preprocessing.

Analisis meliputi:

* jumlah records
* jumlah columns
* data types
* missing values
* duplicate records
* unique identifiers
* relationship antar dataset
* date range
* categorical values
* transaction status

Validasi relationship menunjukkan tidak terdapat orphan records pada foreign-key relationship utama.

---

# 2. Data Preparation

Data kemudian dipersiapkan untuk analisis lebih lanjut.

Proses meliputi:

* parsing date columns
* validasi data type
* validasi unique identifier
* validasi relationship antar tabel
* filtering completed transactions
* validasi transaction amount
* customer-level aggregation

Untuk analisis revenue dan customer value, hanya transaksi dengan status:

```text
completed
```

yang digunakan.

---

# 3. Exploratory Data Analysis

EDA digunakan untuk memahami pola bisnis sebelum membuat machine learning model.

Analisis mencakup:

### Revenue

Mengukur:

* total revenue
* monthly revenue
* revenue trend
* average order value

### Product Category

Mengukur:

* revenue per category
* transaction volume
* quantity sold

### Customer

Menganalisis:

* customer revenue
* purchase frequency
* average order value
* churn behavior
* customer segments

### Engagement

Menganalisis:

* session duration
* pages viewed
* cart additions
* conversion
* bounce rate

### Geographic Performance

Menganalisis:

* customer distribution per country
* revenue per country
* average revenue per customer

---

# 4. Initial Business Insights

Berdasarkan hasil EDA, beberapa temuan utama:

### Revenue

Total completed transaction revenue:

**5.44M**

dengan:

* Completed transactions: **68,700**
* Unique customers with completed transactions: **9,730**
* Average order value: **79.23**

### Category Performance

Kategori dengan revenue terbesar adalah:

1. Electronics
2. Jewelry
3. Automotive
4. Home & Garden
5. Sports

Electronics memberikan kontribusi revenue paling besar dibandingkan kategori lainnya.

### Customer Revenue Concentration

Revenue menunjukkan konsentrasi yang cukup tinggi pada customer dengan nilai pembelian terbesar:

| Customer Group | Revenue Share |
| -------------- | ------------: |
| Top 10%        |        30.45% |
| Top 20%        |        48.73% |
| Top 50%        |        81.57% |

Temuan ini menunjukkan bahwa sebagian besar revenue berasal dari sebagian customer tertentu.

### Churn

Dataset memiliki:

| Status  | Customers |
| ------- | --------: |
| Active  |     8,306 |
| Churned |     1,694 |

Customer churned memiliki rata-rata:

* orders lebih rendah
* revenue lebih rendah

dibandingkan customer active.

### Customer Engagement

Session yang melakukan conversion menunjukkan engagement yang lebih tinggi.

Dibandingkan non-converted sessions:

* average duration lebih tinggi
* average pages viewed lebih tinggi
* cart additions lebih tinggi
* bounce rate lebih rendah

Hal ini menunjukkan adanya hubungan antara engagement dan conversion.

> Hubungan ini digunakan sebagai business observation dan tidak secara otomatis dianggap sebagai causal relationship.

### Country Performance

US merupakan market dengan jumlah customer dan total revenue terbesar.

Beberapa negara lain memiliki average revenue per customer yang relatif tinggi meskipun jumlah customer lebih kecil.

---

# 5. Feature Engineering

Feature engineering dilakukan pada level customer.

## RFM Features

### Recency

Jumlah hari sejak completed transaction terakhir.

```text
recency_days
```

### Frequency

Jumlah completed transactions.

```text
frequency
```

### Monetary

Total revenue yang dihasilkan customer.

```text
monetary
```

---

## Purchase Behavior Features

Feature tambahan:

```text
total_orders
total_quantity
average_order_value
```

---

## Customer Tenure

Mengukur lama customer telah terdaftar:

```text
tenure_days
```

---

## Session Engagement Features

Feature yang berasal dari session:

```text
total_sessions
avg_session_duration
avg_pages_viewed
total_cart_additions
conversion_rate
bounce_rate
```

---

## Purchase Activity

Ditambahkan indikator:

```text
has_purchase
```

Nilai:

```text
1 = memiliki completed transaction
0 = belum memiliki completed transaction
```

270 customer tidak memiliki completed transaction.

Customer tersebut tetap dipertahankan dalam dataset karena tetap merupakan bagian dari customer population dan memiliki target churn.

---

# Final Customer Feature Dataset

Feature dataset yang telah divalidasi:

```text
customer_features.csv
```

Shape:

```text
10,000 rows
21 columns
```

Feature utama:

```text
customer_id
age
gender
country
email_opt_in
has_app
is_churned

recency_days
frequency
monetary

total_orders
total_quantity
average_order_value

tenure_days

total_sessions
avg_session_duration
avg_pages_viewed
total_cart_additions
conversion_rate
bounce_rate

has_purchase
```

Dataset ini menjadi foundation untuk tahap Machine Learning.

---

# 6. Customer Segmentation

Tahap berikutnya menggunakan **RFM + K-Means Clustering**.

Tujuan:

> Mengidentifikasi kelompok customer berdasarkan pola pembelian.

Pipeline:

```text
RFM
 ↓
Handle customers without purchase
 ↓
Feature transformation
 ↓
StandardScaler
 ↓
K-Means
 ↓
Test multiple K
 ↓
Silhouette Score
 ↓
Select K
 ↓
Cluster Profiling
 ↓
Business Segment
```

Jumlah cluster tidak ditentukan secara arbitrer.

Beberapa nilai `K` akan dibandingkan menggunakan:

* Silhouette Score
* cluster size
* cluster characteristics
* business interpretability

Output yang diharapkan:

```text
customer_id
cluster
segment
```

Kemudian setiap cluster akan diprofilkan berdasarkan:

* Recency
* Frequency
* Monetary
* customer count
* revenue contribution

---

# 7. Churn Prediction

Tahap selanjutnya adalah membangun model untuk memprediksi customer yang berpotensi churn.

### Target

```text
is_churned
```

### Candidate Features

Model dapat menggunakan:

```text
age
email_opt_in
has_app
recency_days
frequency
monetary
total_orders
total_quantity
average_order_value
tenure_days
total_sessions
avg_session_duration
avg_pages_viewed
total_cart_additions
conversion_rate
bounce_rate
has_purchase
```

Categorical features seperti:

```text
gender
country
```

akan diproses menggunakan encoding yang sesuai.

### Candidate Models

Model yang dapat dibandingkan:

* Logistic Regression
* Random Forest
* Gradient Boosting

Model dipilih berdasarkan kombinasi:

* Precision
* Recall
* F1-score
* ROC-AUC
* business interpretability

Karena churn prediction merupakan problem bisnis, **accuracy tidak digunakan sebagai satu-satunya metric**.

---

# 8. Model Evaluation

Evaluasi model akan menggunakan:

### Classification Metrics

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
```

### Confusion Matrix

Digunakan untuk memahami:

```text
True Positive
True Negative
False Positive
False Negative
```

Dalam konteks bisnis, False Negative dapat menjadi perhatian karena berarti customer yang sebenarnya berpotensi churn tidak teridentifikasi oleh model.

---

# 9. Business Recommendations

Output akhir project tidak berhenti pada model.

Hasil analytics dan machine learning akan diterjemahkan menjadi rekomendasi seperti:

### High-Value Customers

Customer dengan Monetary dan Frequency tinggi dapat diprioritaskan untuk:

* loyalty program
* personalized offers
* retention campaign

### At-Risk Customers

Customer dengan recency tinggi dan aktivitas yang menurun dapat menjadi target:

* re-engagement campaign
* personalized promotion
* reminder campaign

### Low-Engagement Customers

Customer dengan session engagement rendah dapat dianalisis untuk:

* UX improvement
* targeted campaign
* product recommendation

### Revenue Concentration

Karena sebagian besar revenue berasal dari sebagian customer, perusahaan perlu memperhatikan risiko revenue concentration dan menjaga customer bernilai tinggi.

---

# Technologies

Project menggunakan Python ecosystem:

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

Untuk development:

```text
VS Code
Git
GitHub
```

---

# Data Science Skills Demonstrated

Project ini dirancang untuk menunjukkan kemampuan:

### Data Analysis

* Data cleaning
* Data validation
* Exploratory Data Analysis
* Aggregation
* Statistical analysis

### Feature Engineering

* RFM analysis
* Customer-level aggregation
* Behavioral features
* Engagement features
* Feature validation

### Machine Learning

* Unsupervised learning
* K-Means clustering
* Supervised learning
* Classification
* Model evaluation
* Feature preprocessing

### Business Analytics

* Revenue analysis
* Customer segmentation
* Customer value analysis
* Churn analysis
* Conversion analysis
* Business problem identification
* Data-driven recommendation

---

# Key Deliverables

Final project akan menghasilkan:

```text
1. Clean analytical dataset
2. Customer feature dataset
3. EDA findings
4. Business insights
5. Customer segments
6. Churn prediction model
7. Model evaluation
8. Business recommendations
```

---

# Project Status

| Component                      | Status         |
| ------------------------------ | -------------- |
| Dataset exploration            | ✅ Completed    |
| Data validation                | ✅ Completed    |
| Data preparation               | ✅ Completed    |
| Exploratory Data Analysis      | ✅ Completed    |
| Initial business insights      | ✅ Completed    |
| RFM feature engineering        | ✅ Completed    |
| Purchase behavior features     | ✅ Completed    |
| Customer tenure features       | ✅ Completed    |
| Session engagement features    | ✅ Completed    |
| Final customer feature dataset | ✅ Completed    |
| Customer segmentation          | 🚧 In Progress |
| Churn prediction               | ⏳ Planned      |
| Model evaluation               | ⏳ Planned      |
| Business recommendations       | ⏳ Planned      |
| Final documentation            | ⏳ Planned      |

---

# Conclusion

Project ini bertujuan menunjukkan bagaimana seorang **AI Data Scientist** dapat mengubah raw e-commerce data menjadi:

```text
Data
 ↓
Analysis
 ↓
Insight
 ↓
Features
 ↓
Machine Learning
 ↓
Business Decision
```

Fokus utama bukan hanya mendapatkan model dengan metric tinggi, tetapi menghasilkan **insight yang dapat dipahami dan digunakan oleh business stakeholders**.

---

## Author

**Dzulfidho**

GitHub: `DzuNa76`

---

## Disclaimer

Dataset yang digunakan merupakan **synthetic dataset** untuk tujuan pembelajaran dan portfolio.

Insight dan model yang dihasilkan tidak merepresentasikan kondisi bisnis perusahaan nyata.
