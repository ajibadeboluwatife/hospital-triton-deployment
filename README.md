# hospital-triton-deployment
Model Deployment in Hospitals Using Triton Inference Server.
# 🚑 Hospital Model Deployment with Triton Inference Server

This project demonstrates how to deploy a **hospital patient readmission risk model** using **NVIDIA Triton Inference Server**.

## Features
- Train a hospital risk prediction model (Scikit-Learn → ONNX).
- Deploy the model with Triton inside Docker.
- Optimize inference with batching and GPU acceleration.
- Serve predictions via a simple client API.

## Use Cases
- Predicting hospital readmission risks.
- Supporting clinical decision-making.
- Reducing healthcare costs with optimized ML pipelines.

## Getting Started
```bash
git clone https://github.com/<your-username>/hospital-triton-deployment.git
cd hospital-triton-deployment
bash scripts/run_triton.sh
python scripts/client_inference.py
