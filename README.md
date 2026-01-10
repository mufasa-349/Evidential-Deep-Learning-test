# Evidential AnomalyDINO - Few-Shot Unsupervised Anomaly Detection

Endüstriyel görüntü tabanlı anomali tespiti için evidential deep learning yaklaşımı. Az sayıda normal görüntü ile çalışan, unsupervised (anomali örneği gerektirmeyen) bir sistem.

## 🎯 Proje Özeti

Bu proje, **DINOv2** (Vision Transformer) ve **Memory Bank** yaklaşımını kullanarak, sadece normal görüntülerden öğrenerek anomali tespiti yapar. Evidential Deep Learning ile epistemic uncertainty hesaplayarak, modelin ne kadar emin olduğunu da gösterir.

### Temel Özellikler

- ✅ **Few-shot Learning**: Az sayıda normal görüntü ile çalışır (8 görüntü yeterli)
- ✅ **Unsupervised**: Anomali örneği gerektirmez
- ✅ **Pre-trained Model**: DINOv2 kullanır (transfer learning)
- ✅ **Pixel-level Localization**: Anomali bölgelerini heatmap ile gösterir
- ✅ **Uncertainty Quantification**: Epistemic uncertainty hesaplar

## 🔬 Kullanılan Yöntem

### 1. Feature Extraction (DINOv2)
- Her görüntü 14×14 patch'lere bölünür
- DINOv2 ile her patch 384 boyutlu özellik vektörüne dönüştürülür
- Pre-trained model kullanılır (eğitim gerekmez)

### 2. Memory Bank Oluşturma
- Normal görüntülerden çıkarılan tüm patch embeddings bir "memory bank"te saklanır
- Örnek: 8 normal görüntü → ~11,200 patch embedding

### 3. Anomali Tespiti
- Test görüntüsündeki her patch için memory bank'teki en yakın normal patch bulunur
- Distance değerleri → Evidence → Dirichlet parameters → Uncertainty/Belief maps

### 4. Evidential Inference
- Evidence = exp(-γ × distance²)
- Uncertainty = 2 / (evidence + 2)
- Yüksek uncertainty = Yüksek anomali olasılığı

## 📊 Veri Seti

### Normal Görüntüler (Training)
`data/normal/` klasöründe 8 adet normal fabrika parçası görüntüsü:
- Metal plaka benzeri yapılar
- Düzenli delikler
- Tekstürlü yüzeyler

**Örnek Normal Görüntü:**
```
data/normal/normal_00.png
```

### Test Görüntüleri
`data/test/` klasöründe 12 adet test görüntüsü:
- **6 Normal**: Normal parçalar (test_normal_00.png - test_normal_05.png)
- **6 Anomali**: Anomali içeren parçalar (test_anom_00.png - test_anom_05.png)

**Anomali Türleri:**
- Çizik (scratch)
- Eksik delik (missing hole)
- Ekstra leke (blob)
- Yanık izi (burn mark)

**Örnek Anomali Görüntüsü:**
```
data/test/test_anom_00.png  (Çizik anomali)
data/test/test_anom_02.png  (Eksik delik anomali)
```

## 🚀 Kurulum

### Gereksinimler
```bash
Python 3.8+
```

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd Evidential-Deep-Learning-test
```

2. **Virtual environment oluşturun:**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

### Gerekli Paketler
- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `Pillow` >= 10.0.0
- `numpy` >= 1.24.0
- `faiss-cpu` >= 1.7.4 (opsiyonel, hızlandırma için)

## 💻 Kullanım

### Temel Kullanım

```bash
python evidential_anomalydino.py
```

Script otomatik olarak:
1. `data/normal/` klasöründeki normal görüntülerden memory bank oluşturur
2. `data/test/` klasöründeki test görüntülerini analiz eder
3. Sonuçları `outputs/` klasörüne kaydeder

### Çıktılar

Her test görüntüsü için 3 dosya oluşturulur:

1. **`*_uncertainty.png`**: Epistemic uncertainty heatmap
   - Yüksek değerler (beyaz) = Yüksek anomali olasılığı
   - Düşük değerler (siyah) = Normal bölgeler

2. **`*_belief.png`**: Normal sınıfına olan inanç (belief) map
   - Yüksek değerler = Normal olma inancı yüksek

3. **`*_overlay_uncertainty.png`**: Orijinal görüntü üzerine uncertainty overlay
   - Anomali bölgeleri görsel olarak vurgulanır

4. **`ranking.txt`**: Tüm test görüntülerinin uncertainty skorlarına göre sıralaması

## 📈 Sonuçlar

### Başarılı Senaryo Örnekleri

**Anomali Tespiti:**
- `test_anom_01.png`: Uncertainty score = **1.0000** ✅
- `test_anom_04.png`: Uncertainty score = **1.0000** ✅
- `test_anom_05.png`: Uncertainty score = **1.0000** ✅

**Normal Tespiti:**
- `test_normal_00.png`: Uncertainty score = **0.9920** ✅
- `test_normal_01.png`: Uncertainty score = **0.9906** ✅
- `test_normal_02.png`: Uncertainty score = **0.9933** ✅

### Örnek Output Görselleri

**Anomali Görüntüsü (test_anom_00.png):**
- Uncertainty heatmap: Anomali bölgesi yüksek uncertainty gösterir
- Overlay: Çizik anomali görsel olarak vurgulanır

**Normal Görüntü (test_normal_00.png):**
- Uncertainty heatmap: Düşük ve düzgün dağılım
- Overlay: Anomali belirtisi yok

### Ranking Sonuçları

En yüksek uncertainty skorları anomali görüntülerinde:
```
1.000000  test_anom_01.png  (Anomali ✅)
1.000000  test_anom_04.png  (Anomali ✅)
1.000000  test_anom_05.png  (Anomali ✅)
0.999992  test_anom_02.png  (Anomali ✅)
0.999927  test_anom_00.png  (Anomali ✅)
0.998030  test_anom_03.png  (Anomali ✅)
0.996992  test_normal_03.png  (Normal)
0.995579  test_normal_05.png  (Normal)
...
```

## 🔧 Yapılandırma

`evidential_anomalydino.py` dosyasındaki `Config` sınıfından parametreler ayarlanabilir:

```python
cfg = Config(
    device="cuda" if torch.cuda.is_available() else "cpu",
    image_size=518,           # DINOv2 için önerilen boyut
    patch_size=14,            # ViT-S/14 patch boyutu
    dinov2_model="dinov2_vits14",  # Model versiyonu
    gamma=0.01,                # Evidence mapping parametresi
    agg="max",                 # Image-level score aggregation
    out_dir="outputs",         # Çıktı klasörü
    save_heatmaps=True,        # Heatmap kaydetme
)
```

## 📚 Teknik Detaylar

### Model Mimarisi
- **Backbone**: DINOv2 ViT-S/14 (384 embedding dimension)
- **Memory Bank**: L2 distance tabanlı nearest-neighbor search
- **Evidential Layer**: Dirichlet distribution parametreleri

### Performans
- **Memory Bank Oluşturma**: ~2-3 saniye (8 görüntü)
- **Inference**: ~1-2 saniye/görüntü (CPU)
- **Toplam Test Süresi**: ~15-20 saniye (12 görüntü)

## 🎓 Referanslar

- DINOv2: [Facebook Research](https://github.com/facebookresearch/dinov2)
- Evidential Deep Learning: Few-Shot Unsupervised Anomaly Detection
- Memory Bank yaklaşımı: PatchCore, PaDiM gibi yöntemlerden ilham alınmıştır

## 📝 Notlar

- İlk çalıştırmada DINOv2 modeli otomatik olarak indirilir (~300MB)
- macOS'ta FAISS kullanımı OpenMP çakışması nedeniyle devre dışı (PyTorch fallback aktif)
- GPU kullanımı için CUDA kurulu olmalı ve `device="cuda"` ayarlanmalı

## 👥 Katkıda Bulunanlar

Proje geliştirme aşamasındadır.

---

**Lisans**: MIT (veya belirtilen lisans)
