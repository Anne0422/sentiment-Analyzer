# AI-Powered Sentiment & Topic Tracker (WordPress Integration)

## 📌 Project Introduction
This project is a cross-platform system designed to automate sentiment analysis and topic discovery for WordPress website comments. By bridging a **PHP-based WordPress Plugin** with a **Python Flask API**, the system captures real-time feedback and processes it using NLP (Natural Language Processing). This allows site administrators to visualize the emotional tone of their community through a dynamic dashboard.

---

## 🏗️ Project Structure
The repository is organized into a monorepo structure to separate the frontend CMS logic from the backend analytical engine:

```text
Sentiment-Analyzer-Model/
│
├── 📂 wordpress-plugin/       # Frontend Integration
│   ├── sentiment-tracker.php  # Main Plugin File (API Bridge)
│   ├── css/                   # Dashboard Styling
│   └── js/                    # UI Interactions
│
├── 📂 python-backend/         # NLP Engine
│   ├── app.py                 # Flask API Service
│   ├── requirements.txt       # Dependencies (VADER, Flask, etc.)
│   └── notebook.ipynb         # Google Colab Research & Validation
│
└── README.md                  # Project Documentation
