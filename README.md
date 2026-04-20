# 🌱 GRU-Based Plant Stress Prediction using Time-Series Data

## 📌 Overview
This project focuses on early detection of plant stress using time-series environmental sensor data. A deep learning approach using a GRU (Gated Recurrent Unit) network is implemented and compared with a traditional Random Forest model to evaluate performance in modeling temporal dependencies.

---

## 🎯 Objective
- Predict early-stage plant stress conditions  
- Compare deep learning (GRU) with classical machine learning (Random Forest)  
- Analyze the impact of temporal modeling on prediction performance  

---

## ⚙️ Models Used
- **GRU (Proposed Model)** – captures sequential dependencies in time-series data  
- **Tuned GRU** – optimized version with improved hyperparameters  
- **Random Forest** – baseline model for comparison  

---

## 📊 Input Features
- Temperature  
- Humidity  
- Soil Moisture  

---

## 🧠 Methodology
- Time-series preprocessing using sliding window technique  
- Sequential modeling using GRU neural network  
- Baseline comparison using Random Forest (flattened input)  
- Performance evaluation using standard classification metrics  

---

## 📊 Results

### 🔹 Performance Comparison

| Model                     | Accuracy | Precision | Recall  | F1 Score | Inference Time (s) | Computational Cost |
|--------------------------|----------|----------|--------|----------|--------------------|--------------------|
| Random Forest            | 0.6700   | 0.7111   | 0.9014 | 0.7950   | 0.0419             | Low                |
| GRU (Proposed)           | 0.7755   | 0.7755   | 1.0000 | 0.8736   | 0.3381             | Medium             |
| Tuned GRU (Proposed)     | 0.7653   | 0.7732   | 0.9868 | 0.8670   | 0.1161             | Medium             |

---

### 🔹 Hyperparameter Configuration

| Parameter      | Original GRU | Tuned GRU |
|---------------|-------------|----------|
| GRU Units     | 32          | 64       |
| Batch Size    | 32          | 16       |
| Epochs        | 20          | 30       |
| Learning Rate | 0.001       | 0.001    |

---

## 🔍 Analysis
- GRU significantly outperforms Random Forest in **accuracy, recall, and F1-score**, highlighting its ability to capture temporal dependencies.  
- The tuned GRU improves **inference efficiency** while maintaining strong predictive performance.  
- Random Forest has lower computational cost but lacks the ability to model sequential relationships effectively.  
- High recall values indicate strong capability in detecting plant stress conditions, which is crucial for early intervention in agriculture.  

---

## 🚀 Tech Stack
- Python  
- NumPy, Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

---

## 📂 Project Structure
