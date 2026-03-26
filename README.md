# 🎫 AI Support Ticket Triage System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-green.svg)

An automated Natural Language Processing (NLP) system designed to help IT and Customer Support teams automatically classify and prioritize incoming helpdesk tickets. Built as **Task 1** for the ABC Company virtual internship.

## 📌 The Business Problem
Customer support teams at SaaS companies and IT departments receive hundreds to thousands of tickets every day. The manual triage process—reading each ticket, determining the issue category, and assigning a priority level—is time-consuming and prone to human error. Critical issues often get delayed simply because they are buried at the bottom of the queue.

## 🚀 The Solution
This project implements a Machine Learning decision-support system that:
1. **Automatically categorizes** incoming textual tickets into predefined IT routing departments (e.g., Hardware, Access, HR Support) with **85% accuracy**.
2. **Predicts urgency & priority** levels (High/Medium/Low) based on semantic keywords with **90% accuracy**, ensuring critical business-stoppage events (e.g., server crashes) are immediately flagged.
3. Provides a clean, real-time **Streamlit web application** for interactive testing and demonstration.

---

## 🏗️ Project Architecture & Pipeline

### 1. Data Processing
* **Dataset:** 47,000+ real IT service tickets.
* **Text Normalization:** Lowercasing, punctuation stripping.
* **Stopword Removal & Lemmatization:** Utilizing `NLTK` to strip non-informative context.
* **Feature Extraction:** Conversion of text to numerical vectors using `TfidfVectorizer`.

### 2. Machine Learning Models
* **Algorithms:** Dual `LogisticRegression` models for speed and state-of-the-art text classification capabilities.
* **Serialization:** Models and vectorizers are exported to `models.pkl` using `joblib` for rapid inference in the web frontend.

---

## 📁 Repository Structure

```text
├── data/                                      # Ignored: Raw dataset goes here
├── src/
│   └── train.py                               # 🧠 Core ML pipeline & training script
├── notebooks/
│   └── Support_Ticket_Classification.ipynb    # 📓 Jupyter notebook for EDA
├── app.py                                     # 🌐 Streamlit Web Application
├── models.pkl                                 # 📦 Serialized TF-IDF and LR models
├── requirements.txt                           # 🐍 Python package dependencies
└── README.md                                  # 📄 Project documentation
```

---

## 💻 Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **NLP Libraries:** NLTK, spaCy, Regular Expressions (re)
- **Frontend / Deployment:** Streamlit

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/sneharajappan/Support_Ticket_Classification.git
cd Support_Ticket_Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
```bash
streamlit run app.py
```
*The application should automatically open in your default web browser at `http://localhost:8501`.*

### 4. Retrain Models (Optional)
If you wish to explore the data pipeline or supply a new `.csv` dataset, you can retrain the `.pkl` files:
```bash
python src/train.py
```

---

*This project was developed by Sneha Rajappan.*
