# Starbucks Mobile App Market Segmentation - Clean Code

## 📋 Deskripsi

Repository ini berisi clean code untuk analisis segmentasi pelanggan Starbucks Mobile App menggunakan:
- **RFM Analysis** (Recency, Frequency, Monetary)
- **K-Means Clustering**
- **PCA** (Principal Component Analysis)

## 🏗️ Struktur Kode

### 1. **Configuration & Constants** (`Config`)
- Menyimpan semua konstanta dan konfigurasi di satu tempat
- Mudah untuk dimodifikasi tanpa mengubah logic

### 2. **DataLoader**
Menangani loading dan preprocessing data:
- `load_data()`: Load CSV file
- `filter_mobile_app()`: Filter data Mobile App
- `convert_date_column()`: Convert tanggal ke datetime

### 3. **EDA (Exploratory Data Analysis)**
Visualisasi dan analisis data:
- `print_data_summary()`: Summary statistik
- `plot_continuous_distributions()`: KDE plot
- `plot_discrete_distributions()`: Histogram
- `plot_correlation_matrix()`: Correlation heatmap
- `plot_boxplots()`: Deteksi outliers

### 4. **FeatureEngineering**
Transformasi fitur:
- `encode_ordinal()`: Encoding kategorikal
- `create_rfm_features()`: Membuat RFM features
- `scale_features()`: Standardisasi
- `apply_pca()`: Reduksi dimensi

### 5. **ClusteringAnalysis**
Clustering dan evaluasi:
- `find_optimal_clusters_elbow()`: Elbow method
- `find_optimal_clusters_silhouette()`: Silhouette analysis
- `perform_clustering()`: K-Means clustering
- `get_cluster_centers()`: Extract cluster centers

### 6. **StarbucksSegmentation (Main Pipeline)**
Orchestrator utama yang menjalankan seluruh pipeline

## 🚀 Cara Menggunakan

### Instalasi Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Menjalankan Pipeline Lengkap

```python
from starbucks_segmentation_clean import StarbucksSegmentation

# Initialize
segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')

# Run full pipeline (auto-select optimal clusters)
cluster_centers = segmentation.run_pipeline(auto_select_clusters=True)

# Atau manual specify jumlah cluster
cluster_centers = segmentation.run_pipeline(auto_select_clusters=False, n_clusters=3)
```

### Menjalankan Step by Step

```python
# Step 1: Load data
segmentation.load_and_prepare_data()

# Step 2: EDA
segmentation.explore_data()

# Step 3: Feature Engineering
segmentation.engineer_features()

# Step 4: Scaling & PCA
segmentation.scale_and_reduce(n_components=3)  # atau None untuk auto

# Step 5: Find optimal clusters
optimal_k = segmentation.find_optimal_clusters()

# Step 6: Final clustering
cluster_centers = segmentation.perform_clustering(n_clusters=optimal_k)
```

### Menggunakan Individual Components

```python
from starbucks_segmentation_clean import DataLoader, EDA, ClusteringAnalysis

# Load data
df = DataLoader.load_data('data.csv')

# EDA
EDA.print_data_summary(df)
EDA.plot_correlation_matrix(df)

# Clustering
wss_df = ClusteringAnalysis.find_optimal_clusters_elbow(df)
```

## 📊 Output

Pipeline akan menghasilkan:
1. **Cluster Centers** dalam format DataFrame
2. **Visualisasi** Elbow & Silhouette
3. **Metrics** untuk setiap tahap

Contoh output:
```
     cluster  recency  frequency   monetary  rewards_member  order_ahead
0  Cluster 1    45.23      12.45     234.56            0.85         0.65
1  Cluster 2    78.90       5.32     123.45            0.45         0.35
2  Cluster 3    23.45      25.67     456.78            0.95         0.85
```

## ✨ Perbaikan dari Kode Original

### 1. **Modular & Reusable**
- Fungsi-fungsi dipisahkan ke dalam class yang logis
- Bisa digunakan untuk dataset lain dengan minimal modification

### 2. **Documentation**
- Setiap fungsi memiliki docstring
- Type hints untuk semua parameter
- Komentar yang jelas

### 3. **Error Handling**
- Try-catch untuk file loading
- Validasi input parameters
- Informative error messages

### 4. **Naming Convention**
- Konsisten menggunakan English
- Descriptive variable names
- PEP 8 compliant

### 5. **Best Practices**
- DRY (Don't Repeat Yourself) principle
- Single Responsibility Principle
- Configuration separated from logic
- Proper use of dataclasses

### 6. **Maintainability**
- Easy to add new features
- Easy to modify constants
- Clear separation of concerns

### 7. **Performance**
- Efficient data operations
- Proper use of pandas methods
- Memory-conscious operations

## 🔧 Customization

### Mengubah Konfigurasi

```python
config = Config()
config.RANDOM_STATE = 123
config.DISCRETE_COLS = ['custom_col1', 'custom_col2']
```

### Menambah Custom Analysis

```python
class CustomAnalysis:
    @staticmethod
    def my_analysis(df):
        # Your analysis here
        pass

# Gunakan dalam pipeline
segmentation.custom_step = CustomAnalysis.my_analysis(segmentation.data)
```

## 📝 Notes

- Data harus dalam format CSV
- Kolom yang dibutuhkan: customer_id, order_id, order_date, total_spend, dll.
- Default menggunakan automatic optimal cluster selection
- Semua visualisasi bisa di-toggle on/off

## 🤝 Kontribusi

Untuk improvement atau bug fixes:
1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Create Pull Request

## 📄 License

MIT License - Feel free to use and modify

---

**Author**: Arnold Orlando  
**Version**: 2.0 (Clean Code)  
**Last Updated**: 2026
