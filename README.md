# Spiking transformer with learnable threshold mechanism for underwater image dehazing to aid vision-based navigation

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-SNN-red)
![Journal](https://img.shields.io/badge/Journal-NCE%202026-orange)

**Published in Neuromorphic Computing and Engineering (2026)**

This repository contains the implementation of **snnTrans-DHZ**, a lightweight spiking-transformer framework for **underwater image dehazing**. The method introduces a **learnable threshold membrane potential mechanism** to improve temporal dynamics and memory retention in spiking neural processing, while preserving computational efficiency for deployment-oriented vision systems.

The framework is designed to address underwater image degradation caused by haze, scattering, absorption, color distortion, and reduced visibility. It combines **spiking neural computation**, **transformer-based feature extraction**, and **hybrid RGB-LAB processing** to produce visibility-enhanced underwater images suitable for downstream robotic perception and vision-based navigation.

---

## Key Contributions

- **🧠 Learnable threshold spiking neurons:** Introduces **LTMP-LIF** neurons, enabling adaptive threshold membrane potential dynamics during backpropagation for improved temporal modeling.

- **🌊 Underwater image dehazing with spiking transformers:** Proposes **snnTrans-DHZ**, a lightweight transformer-integrated SNN architecture tailored to underwater image enhancemnet.

- **🎨 Hybrid RGB-LAB processing:** Uses joint RGB-LAB feature representations to better separate luminance and chromaticity information for improved haze removal and color recovery.

- **📉 Lightweight and efficient:** Achieves comparable performance with only **0.5670M parameters**, **7.42 GSOPs**, and **0.151 J** energy consumption.

- **📊 Strong benchmark results:** Evaluated on **UIEB** and **EUVP**, achieving competitive restoration quality with improved energy efficiency.

- **🤖 Robotics relevance:** Suitable for underwater robotics, environmental monitoring, and vision-based navigation in degraded marine environments.

---

## System Overview

<p align="center"> <img src="assets/architecture.png" width="88%"> </p> <p align="center"> <em>Overview of the proposed snnTrans-DHZ architecture.</em> </p>

The proposed **snnTrans-DHZ** architecture is built around three main modules:

- **K Estimator module:** extracts features from different color space representations.  
- **Background light estimator module:** jointly estimates the background light component from RGB-LAB features.  
- **Soft image reconstruction module:** reconstructs haze-free, visibility-enhanced underwater images.  

Underwater images are first transformed into **time-dependent sequences**, then represented in **LAB color space**, and finally processed through the spiking-transformer framework using **surrogate gradient-based backpropagation through time**.

## Efficiency

- **Parameters:** 0.5670M  
- **Computational cost:** 7.42 GSOPs  
- **Energy consumption:** 0.151 J  
- **Energy efficiency improvement:** 3.3× over the lightest transformer-based state-of-the-art baseline  

<p align="center">
  <img src="assets/quantitative_tables.jpg" width="78%">
</p>
<p align="center">
  <em>Performance evaluation with state-of-the-art spiking based underwater image enhancement method.</em>
</p>

## Qualitative Results

<p align="center">
  <img src="assets/qualitative_results.png" width="92%">
</p>
<p align="center">
  <em>Qualitative comparison on various underwater scenes.</em>
</p>

---

## Repository Structure

```bash
.
├── checkpoints/           # Saved model checkpoints
├── data/                  # Training / testing data
├── dehazed_images/        # Output restored images
├── .gitattributes
├── dataset_rgblab.py      # Dataset preparation 
├── README.md
├── snntrans_model.py      # Model architecture
├── snntrans_test.py       # Inference script
└── snntrans_train.py      # Training script

## Getting Started

For installation, setup, and usage, please use the training and testing scripts provided in the repository.

### Requirements

- Python 3.8+
- PyTorch
- Additional dependencies listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt

Training
