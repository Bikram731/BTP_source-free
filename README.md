# Source-Free Cross-Machine Fault Diagnosis (SFDA)

**A Two-Stage Pseudo-Supervised Framework with FFT Signal Preprocessing**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange) ![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
[cite_start]This repository contains the implementation of a **Source-Free Domain Adaptation (SFDA)** framework for industrial fault diagnosis[cite: 965]. [cite_start]The goal is to diagnose faults (Healthy, Chipped, Cracked, Missing Tooth) on a target machine **without accessing the original source training data**, addressing critical data privacy (GDPR) and bandwidth constraints in Industry 4.0[cite: 991, 992].

[cite_start]The core algorithm is based on **Two-Stage Pseudo-Supervised Learning**, enhanced with a custom **Fast Fourier Transform (FFT) preprocessing pipeline** to resolve sensor modality conflicts and isolate Gear Mesh Frequencies[cite: 1045, 1052].

### Key Features
* [cite_start]**Source-Free Adaptation:** Adapts a pre-trained model to a new machine using only unlabeled target data[cite: 986].
* [cite_start]**FFT Preprocessing:** Converts noisy time-domain signals into frequency-domain spectral peaks to eliminate negative transfer[cite: 1050].
* **Two-Stage Learning:**
    1.  [cite_start]**Alignment:** Uses Semi-Supervised Clustered KNN (SSCKNN) to generate initial pseudo-labels[cite: 1013].
    2.  [cite_start]**Refinement:** Applies Adaptive Thresholding and Contrastive Learning to filter noise[cite: 1014].
* [cite_start]**High Performance:** Achieved **89.22% accuracy** (Unsupervised), recovering **97%** of the supervised upper bound performance.

---

## 📂 Repository Structure

```text
├── data/                        # Dataset folder (Place SEU dataset here)
├── 01_train_SEU_source_model.py # Step 1: Train the source model (Teacher)
├── 02_prepare_target_data.py    # Step 2: Preprocess target data
├── 04_emergency_benchmark_fft.py   # Step 3: Establish Supervised Upper Bound
├── 06_proof_with_fft.py         # Step 4: Run Main Adaptation (The Core Algorithm)
├── 08_visualize_results.py      # Step 5: Generate t-SNE & Confusion Matrix
├── requirements.txt             # Dependencies
└── README.md
