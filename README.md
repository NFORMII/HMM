# Human Activity Recognition using Hidden Markov Models

> **Formative 2**
>  **Machine Learning Techniques 2*  


---

##  Overview

This project implements a **Gaussian Hidden Markov Model (HMM)** to classify human physical activities from smartphone sensor data. Using accelerometer and gyroscope signals recorded with the **Sensor Logger** app, the model learns to distinguish between four activities:

| Activity | Description |
|---|---|
|  **Still** | Phone placed flat on a surface, no movement |
|  **Standing** | Phone held at waist level, natural body sway |
|  **Walking** | Consistent walking pace |
|  **Jumping** | Continuous jumping |

The model achieves **77.8% overall accuracy** on unseen test data, with near-perfect classification of jumping (98.1%) and walking (92.2%).

---

##  Repository Structure

```
HMM/
├── data/
│   ├── still/          # 50 labelled CSV files
│   ├── standing/       # 50 labelled CSV files
│   ├── walking/        # 52 labelled CSV files
│   └── jumping/        # 50 labelled CSV files
├── zips/               # Raw Sensor Logger zip exports
├── hmm_activity_recognition.ipynb   # Main notebook
├── evaluation_metrics.csv           # Per-activity metrics
├── figures/
│   ├── fig_raw_signals.png
│   ├── fig_normalisation.png
│   ├── fig_convergence.png
│   ├── fig_transition_matrix.png
│   ├── fig_emission_means.png
│   ├── fig_confusion_matrix.png
│   ├── fig_decoded_sequence.png
│   └── fig_dashboard.png
├── report.pdf
└── README.md
```

---

## Data Collection

- **App:** Sensor Logger (iOS/Android)
- **Sensors:** Accelerometer + Gyroscope
- **Sampling rate:** 100 Hz (10 ms interval)
- **Files:** 202 total (50–52 per activity)
- **Format:** ZIP archives → merged CSV (timestamp, ax, ay, az, gx, gy, gz)

### Sampling Rates

| Device | Sampling Rate |
|---|---|
| iPhone 8 | 100 Hz |


All recordings were resampled to a harmonised **100 Hz** target rate using `scipy.signal.resample`.

---

## Methodology

###  Preprocessing
- Zip extraction and CSV merging (Accelerometer + Gyroscope merged on timestamp)
- Sliding window segmentation: **1-second windows (100 samples)**, 50% overlap

###  Feature Extraction (25 features per window)

| Domain | Features |
|---|---|
| Time | Mean, Std Dev, RMS — per axis (ax, ay, az, gx, gy, gz) |
| Time | Signal Magnitude Area (SMA) |
| Frequency | Dominant frequency, Spectral energy  per accelerometer axis |

###  Normalisation
Z-score normalisation (StandardScaler) applied to all features justified by differing axis scales and units.

###  HMM Architecture

| Component | Definition |
|---|---|
| Hidden States | 4 activities: still, standing, walking, jumping |
| Observations | 25-dimensional feature vectors |
| Emission | Gaussian (diagonal covariance) |
| Training | Baum–Welch (EM), tol=1e-4, max 300 iterations |
| Decoding | Viterbi algorithm |

---

## Results

### Overall Accuracy: **77.8%**

| Activity | N Samples | Sensitivity | Specificity |
|---|---|---|---|
| Still | 64 | 65.6% | 86.5% |
| Standing | 45 | 55.6% | 86.8% |
| Walking | 51 | 92.2% | 99.4% |
| Jumping | 52 | 98.1% | 97.5% |

### Key Findings
- **Jumping** and **Walking** are classified near-perfectly due to distinct frequency and amplitude signatures
- **Still vs Standing** is the primary source of error  both activities share similar low-variance, low-energy profiles
- The Baum–Welch algorithm converged in **52 iterations** (log-likelihood: 85,544.61)
- Transition matrix shows realistic self-persistence: jumping (0.962), still (0.941), walking (0.913), standing (0.861)

---

## How to Run

### Prerequisites
```bash
pip install hmmlearn scikit-learn numpy pandas scipy matplotlib seaborn
```

### Steps
1. Clone the repository
2. Place all `.zip` recordings in the `zips/` folder
3. Open `hmm_activity_recognition.ipynb` in Jupyter or VS Code
4. Run **Cell 0** (install hmmlearn if needed):
   ```python
   import subprocess, sys
   subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
   ```
5. Run all cells top to bottom

---

##  Task Allocation

| Task | Member |
|---|---|
| Data collection (still, standing) | Member 1 |
| Data collection (walking, jumping) | Member 2 |
| Data preprocessing & unzip pipeline | Member 1 |
| Feature extraction (time domain) | Member 2 |
| Feature extraction (frequency domain) | Member 1 |
| HMM implementation & Baum–Welch | Member 2 |
| Viterbi decoding | Member 1 |
| Evaluation & metrics | Member 2 |
| Visualisations & dashboard | Member 1 |
| Report writing | Member 2 |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `hmmlearn` | ≥0.3.0 | Gaussian HMM, Baum–Welch, Viterbi |
| `scikit-learn` | ≥1.3.0 | Normalisation, metrics, confusion matrix |
| `numpy` | ≥1.24 | Numerical operations |
| `pandas` | ≥2.0 | CSV loading and merging |
| `scipy` | ≥1.10 | FFT, resampling |
| `matplotlib` | ≥3.7 | Visualisations |
| `seaborn` | ≥0.12 | Heatmaps |

---

