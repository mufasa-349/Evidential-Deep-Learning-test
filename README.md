# Evidential AnomalyDINO - Few-Shot Unsupervised Anomaly Detection

Industrial image-based anomaly detection using evidential deep learning approach. A system that works with a small number of normal images, unsupervised (does not require anomaly examples).

## 🎯 Project Summary

This project uses **DINOv2** (Vision Transformer) and **Memory Bank** approach to detect anomalies by learning only from normal images. It also shows how confident the model is by calculating epistemic uncertainty through Evidential Deep Learning.

### Key Features

- ✅ **Few-shot Learning**: Works with a small number of normal images (8 images sufficient)
- ✅ **Unsupervised**: Does not require anomaly examples
- ✅ **Pre-trained Model**: Uses DINOv2 (transfer learning)
- ✅ **Pixel-level Localization**: Shows anomaly regions with heatmaps
- ✅ **Uncertainty Quantification**: Calculates epistemic uncertainty

## 🔬 Methodology

### 1. Feature Extraction (DINOv2)
- Each image is divided into 14×14 patches
- Each patch is converted to a 384-dimensional feature vector using DINOv2
- Pre-trained model is used (no training required)

### 2. Memory Bank Construction
- All patch embeddings extracted from normal images are stored in a "memory bank"
- Example: 8 normal images → ~11,200 patch embeddings

### 3. Anomaly Detection
- For each patch in the test image, the nearest normal patch in the memory bank is found
- Distance values → Evidence → Dirichlet parameters → Uncertainty/Belief maps

### 4. Evidential Inference
- Evidence = exp(-γ × distance²)
- Uncertainty = 2 / (evidence + 2)
- High uncertainty = High anomaly probability

## 📊 Dataset

### Normal Images (Training)
8 normal factory part images in `data/normal/` folder:
- Metal plate-like structures
- Regular holes
- Textured surfaces

**Example Normal Image:**
```
data/normal/normal_00.png
```

### Test Images
12 test images in `data/test/` folder:
- **6 Normal**: Normal parts (test_normal_00.png - test_normal_05.png)
- **6 Anomaly**: Parts containing anomalies (test_anom_00.png - test_anom_05.png)

**Anomaly Types:**
- Scratch
- Missing hole
- Extra blob
- Burn mark

**Example Anomaly Images:**
```
data/test/test_anom_00.png  (Scratch anomaly)
data/test/test_anom_02.png  (Missing hole anomaly)
```

## 🚀 Installation

### Requirements
```bash
Python 3.8+
```

### Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Evidential-Deep-Learning-test
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install packages:**
```bash
pip install -r requirements.txt
```

### Required Packages
- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `Pillow` >= 10.0.0
- `numpy` >= 1.24.0
- `faiss-cpu` >= 1.7.4 (optional, for acceleration)

## 💻 Usage

### Basic Usage

```bash
python evidential_anomalydino.py
```

The script automatically:
1. Creates a memory bank from normal images in `data/normal/` folder
2. Analyzes test images in `data/test/` folder
3. Saves results to `outputs/` folder

### Outputs

For each test image, 3 files are created:

1. **`*_uncertainty.png`**: Epistemic uncertainty heatmap
   - High values (white) = High anomaly probability
   - Low values (black) = Normal regions

2. **`*_belief.png`**: Belief map for normal class
   - High values = High confidence in being normal

3. **`*_overlay_uncertainty.png`**: Uncertainty overlay on original image
   - Anomaly regions are visually highlighted

4. **`ranking.txt`**: Ranking of all test images by uncertainty scores

## 📈 Results

### Successful Scenario Examples

**Anomaly Detection:**
- `test_anom_01.png`: Uncertainty score = **1.0000** ✅
- `test_anom_04.png`: Uncertainty score = **1.0000** ✅
- `test_anom_05.png`: Uncertainty score = **1.0000** ✅

**Normal Detection:**
- `test_normal_00.png`: Uncertainty score = **0.9920** ✅
- `test_normal_01.png`: Uncertainty score = **0.9906** ✅
- `test_normal_02.png`: Uncertainty score = **0.9933** ✅

### Example Output Visualizations

**Anomaly Image (test_anom_00.png):**
- Uncertainty heatmap: Anomaly region shows high uncertainty
- Overlay: Scratch anomaly is visually highlighted

**Normal Image (test_normal_00.png):**
- Uncertainty heatmap: Low and uniform distribution
- Overlay: No anomaly indication

### Ranking Results

Highest uncertainty scores are in anomaly images:
```
1.000000  test_anom_01.png  (Anomaly ✅)
1.000000  test_anom_04.png  (Anomaly ✅)
1.000000  test_anom_05.png  (Anomaly ✅)
0.999992  test_anom_02.png  (Anomaly ✅)
0.999927  test_anom_00.png  (Anomaly ✅)
0.998030  test_anom_03.png  (Anomaly ✅)
0.996992  test_normal_03.png  (Normal)
0.995579  test_normal_05.png  (Normal)
...
```

## 🔧 Configuration

Parameters can be adjusted from the `Config` class in `evidential_anomalydino.py`:

```python
cfg = Config(
    device="cuda" if torch.cuda.is_available() else "cpu",
    image_size=518,           # Recommended size for DINOv2
    patch_size=14,            # ViT-S/14 patch size
    dinov2_model="dinov2_vits14",  # Model version
    gamma=0.01,                # Evidence mapping parameter
    agg="max",                 # Image-level score aggregation
    out_dir="outputs",         # Output folder
    save_heatmaps=True,        # Save heatmaps
)
```

## 📚 Technical Details

### Model Architecture
- **Backbone**: DINOv2 ViT-S/14 (384 embedding dimension)
- **Memory Bank**: L2 distance-based nearest-neighbor search
- **Evidential Layer**: Dirichlet distribution parameters

### Performance
- **Memory Bank Construction**: ~2-3 seconds (8 images)
- **Inference**: ~1-2 seconds/image (CPU)
- **Total Test Time**: ~15-20 seconds (12 images)

## 🎓 References

- DINOv2: [Facebook Research](https://github.com/facebookresearch/dinov2)
- Evidential Deep Learning: Few-Shot Unsupervised Anomaly Detection
- Memory Bank approach: Inspired by methods like PatchCore, PaDiM

## 📝 Notes

- DINOv2 model is automatically downloaded on first run (~300MB)
- FAISS usage is disabled on macOS due to OpenMP conflicts (PyTorch fallback active)
- For GPU usage, CUDA must be installed and `device="cuda"` should be set

-----
**License**: 
