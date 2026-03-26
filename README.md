# Support Ticket Classification & Prioritization System

This machine learning project helps IT Customer Support resolve issues faster by automatically categorizing tickets and prioritizing the urgency of requests using Natural Language Processing (NLP).

## Project Structure

```text
c:\Intern-2\ticket_classification\
├── data/
│   └── all_tickets_processed_improved_v3.csv  # The IT Service Ticket dataset
├── src/
│   └── train.py           # 🧠 Core Machine Learning Training Script
├── notebooks/
│   └── Support_Ticket_Classification.ipynb    # Jupyter Notebook version of the ML code
├── app.py                 # 🌐 Streamlit Web Application Frontend
├── models.pkl             # Serialized trained model and vectorizer
└── requirements.txt       # Python dependencies
```

## Where is the Machine Learning Code?
Your core Machine Learning code is located in two places depending on how you prefer to read it:
1. **Python Script:** `src/train.py` (Used for training the production model)
2. **Jupyter Notebook:** `notebooks/Support_Ticket_Classification.ipynb` (Created for Jupyter Notebook exploration)

Both files contain the data preprocessing, `TfidfVectorizer` setup, and the `LogisticRegression` models for categorization and priority.

## How to Run

**1. Train the Models (Optional, already trained):**
```bash
python src/train.py
```

**2. Run the Web Application:**
```bash
streamlit run app.py
```
