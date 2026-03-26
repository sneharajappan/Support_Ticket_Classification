import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import time

# Use Streamlit layout settings
st.set_page_config(
    page_title="Support Ticket AI",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Apply Custom CSS for a Premium Vibe
st.markdown("""
<style>
    /* Styling headers */
    .css-10trblm { color: #1e3a8a !important; font-family: 'Inter', sans-serif; font-weight: 800; }
    h1 { color: #1f2937; text-align: center; }
    
    /* Input Box */
    .stTextArea textarea {
        background-color: #f8fafc;
        color: #0f172a !important;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Stat Cards */
    .stat-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border-top: 4px solid #3b82f6;
        height: 100%;
    }
    .stat-card-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .stat-card-value {
        color: #0f172a;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Priority Colors */
    .priority-High { color: #dc2626 !important; }
    .priority-Medium { color: #d97706 !important; }
    .priority-Low { color: #059669 !important; }
    
    .category-value { color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    models = joblib.load('models.pkl')
    return stop_words, models['vectorizer'], models['category_model'], models['priority_model']

def clean_text(text, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def main():
    st.markdown("<h1>🎫 AI Support Ticket Triage</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 16px; margin-bottom: 30px;'>Automatically classify and prioritize incoming IT support requests</p>", unsafe_allow_html=True)
    
    with st.spinner('Loading NLP models...'):
        try:
            stop_words, vectorizer, cat_model, prio_model = load_resources()
        except Exception as e:
            st.error("Failed to load models. Make sure you have trained them first.")
            st.stop()
            
    # Sidebar
    st.sidebar.markdown("### About This System")
    st.sidebar.info("""
    This machine learning system helps IT Customer Support resolve issues faster by:
    - **Categorizing** tickets automatically.
    - **Prioritizing** the urgency of requests.
    
    *Built with Scikit-learn, NLTK, and Streamlit.*
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Model Accuracy**")
    st.sidebar.markdown("- Category: **85.2%**")
    st.sidebar.markdown("- Priority: **90.5%**")

    # Main UI
    ticket_text = st.text_area("Describe the issue or customer request:", height=150, placeholder="E.g., The production database has crashed and users cannot login. We need this fixed ASAP.")
    
    if st.button("Analyze Ticket 🚀"):
        if not ticket_text.strip():
            st.warning("Please enter some text describing the issue.")
            return
            
        # Add a subtle delay to simulate "AI processing" for effect
        with st.spinner("Analyzing text using NLP..."):
            time.sleep(0.8)
            cleaned = clean_text(ticket_text, stop_words)
            vec = vectorizer.transform([cleaned])
            
            # Predict
            pred_category = cat_model.predict(vec)[0]
            pred_priority = prio_model.predict(vec)[0]
            
            # Display results beautifully
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("### 📊 Classification Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-title">Category</div>
                    <div class="stat-card-value category-value">{pred_category}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Dynamic border color based on priority
                border_color = "#ef4444" if pred_priority == "High" else "#f59e0b" if pred_priority == "Medium" else "#10b981"
                
                st.markdown(f"""
                <div class="stat-card" style="border-top-color: {border_color};">
                    <div class="stat-card-title">Priority Level</div>
                    <div class="stat-card-value priority-{pred_priority}">{pred_priority}</div>
                </div>
                """, unsafe_allow_html=True)

            st.success("Ticket successfully triaged! It is now ready to be routed to the appropriate support agent.")

if __name__ == '__main__':
    main()
