---
title: Deepfake Detection System
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# 🔍 Deepfake Detection System

A full-stack AI web application that detects **Deepfake , Real and AI-generated media** from images and videos using deep learning and forensic analysis.

Built with **Flask, TensorFlow (EfficientNetB0), and OpenCV**, this system combines ML predictions with low-level visual signals to deliver explainable results.

---

## 🚀 Overview

This project goes beyond simple classification. It analyzes media using:

- A trained CNN model (EfficientNetB0)
- Frame-by-frame video inspection
- Multiple forensic signals (noise, texture, lighting, frequency patterns)

The goal is not just detection — but **understanding *why* something is fake**.

---

## ✨ Key Features

### 🧠 Detection Engine
- Classifies media into:
  - `FAKE` (Deepfake / face swap)
  - `AI-GENERATED`
  - `AUTHENTIC`
- Image and video support
- Frame sampling for videos
- Confidence scoring per frame

---

### 🔬 Forensic Analysis
Uses OpenCV-based signals such as:
- Texture inconsistency
- Noise irregularities
- Edge density anomalies
- Lighting mismatch
- Frequency (DCT) patterns
- Color saturation imbalance

Each result includes **interpretable evidence**, not just a label.

---

### 📊 User Experience
- Clean dashboard with detection history
- Confidence visualization
- Detailed result breakdown
- PDF report download
- Dark mode UI

---

### 🔐 Authentication & Security
- JWT-based login system
- Password hashing (bcrypt)
- Password reset via email
- File validation using actual content (not just extensions)
- Rate limiting on APIs

---

### 👨‍💼 Admin Panel
- View all users
- Monitor detection activity
- Access system-wide history
- Manage users

---

## 🛠️ Tech Stack

**Backend**
- Flask
- Flask-SQLAlchemy
- Flask-Limiter
- Flask-Mail

**Machine Learning**
- TensorFlow / Keras
- EfficientNetB0
- OpenCV
- NumPy

**Frontend**
- HTML, CSS, JavaScript
- Chart.js
- jsPDF

**Database**
- SQLite (development)
- PostgreSQL (recommended for production)

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
````

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file:

```env
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret

ADMIN_EMAIL=admin@deepfake.com
ADMIN_PASSWORD=StrongPassword123!

MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password

UPLOAD_RETENTION_DAYS=7
```

### 4. Run the application

```bash
python app.py
```

Open:
👉 [http://localhost:5000](http://localhost:5000)

---

## 🤖 Model Training (Optional)

If the trained model is not present, the app runs in **demo mode**.

### Dataset

Download **Celeb-DF-v2** and place it outside the project directory.

### Train

```bash
python train_model.py --epochs 20 --batch-size 16
```

Model will be saved in:

```
ml_models/cnn_model.h5
```

---

## ⚠️ Limitations

* Video processing is synchronous (can be slow)
* SQLite not suitable for multi-user scaling
* No Docker setup yet

---

## 💡 Future Improvements

* Async processing with Celery
* Real-time detection API
* Model performance optimization
* Docker deployment

---

## 📄 License

MIT License

---

## 🙌 Credits

* Celeb-DF-v2 Dataset
* EfficientNet (Google Research)
* OpenCV

