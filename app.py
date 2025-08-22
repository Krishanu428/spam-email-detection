import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data only once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Set page config
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 20px;
    }
    .spam-text {
        color: #ff4b4b;
        font-weight: bold;
    }
    .ham-text {
        color: #00b074;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.ps.stem(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    def train_model(self, X, y, model_type='naive_bayes'):
        X_processed = [self.preprocess_text(text) for text in X]
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        
        self.model.fit(X_vectorized, y)
        return X_vectorized, y
    
    def predict(self, text):
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(vectorized_text)
        probability = self.model.predict_proba(vectorized_text)
        return prediction[0], probability[0]
    
    def save_model(self, filename):
        joblib.dump({'vectorizer': self.vectorizer, 'model': self.model}, filename)
    
    def load_model(self, filename):
        data = joblib.load(filename)
        self.vectorizer = data['vectorizer']
        self.model = data['model']

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def create_visualizations(df, classifier):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", "Category Distribution", "Word Clouds", 
        "Text Length Analysis", "Model Performance"
    ])
    
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.write("Dataset Information:")
            st.write(f"Total messages: {len(df)}")
            st.write(f"Spam messages: {len(df[df['Category'] == 'spam'])}")
            st.write(f"Ham messages: {len(df[df['Category'] == 'ham'])}")
            st.write(f"Missing values: {df.isnull().sum().sum()}")
    
    with tab2:
        st.subheader("Category Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names='Category', title='Spam vs Ham Distribution',
                         color='Category', color_discrete_map={'ham': '#00b074', 'spam': '#ff4b4b'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            category_counts = df['Category'].value_counts()
            fig = go.Figure(data=[go.Bar(x=category_counts.index, y=category_counts.values,
                                         marker_color=['#00b074', '#ff4b4b'])])
            fig.update_layout(title='Spam vs Ham Counts', xaxis_title='Category', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)
        with col1:
            spam_text = ' '.join(df[df['Category'] == 'spam']['Message'].astype(str))
            if spam_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(spam_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
        with col2:
            ham_text = ' '.join(df[df['Category'] == 'ham']['Message'].astype(str))
            if ham_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(ham_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
    
    with tab4:
        st.subheader("Text Length Analysis")
        df['Message_Length'] = df['Message'].astype(str).apply(len)
        fig = px.histogram(df, x='Message_Length', color='Category',
                           title='Message Length Distribution',
                           marginal='box', nbins=50,
                           color_discrete_map={'ham': '#00b074', 'spam': '#ff4b4b'})
        st.plotly_chart(fig, use_container_width=True)
        avg_length = df.groupby('Category')['Message_Length'].mean().reset_index()
        fig = px.bar(avg_length, x='Category', y='Message_Length',
                     title='Average Message Length by Category',
                     color='Category', color_discrete_map={'ham': '#00b074', 'spam': '#ff4b4b'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Model Performance Analysis")
        if classifier.model is not None:
            X = df['Message'].astype(str)
            y = df['Category']
            if len(df) < 10:
                st.warning("Not enough data to evaluate model performance (min 10 samples).")
                return
            class_counts = y.value_counts()
            try:
                if all(class_counts >= 2):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    st.warning("Some classes have <2 samples. Using random split instead of stratified.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                X_test_processed = [classifier.preprocess_text(text) for text in X_test]
                X_test_vectorized = classifier.vectorizer.transform(X_test_processed)
                y_pred = classifier.model.predict(X_test_vectorized)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label='spam', zero_division=0)
                recall = recall_score(y_test, y_pred, pos_label='spam', zero_division=0)
                f1 = f1_score(y_test, y_pred, pos_label='spam', zero_division=0)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precision", f"{precision:.2%}")
                col3.metric("Recall", f"{recall:.2%}")
                col4.metric("F1 Score", f"{f1:.2%}")
                cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
                fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['Ham', 'Spam'], y=['Ham', 'Spam'], title="Confusion Matrix",
                                color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            except Exception as e:
                st.error(f"Evaluation error: {str(e)}")

def main():
    st.markdown('<h1 class="main-header">📧 Spam Email Detection</h1>', unsafe_allow_html=True)
    if 'classifier' not in st.session_state:
        st.session_state.classifier = SpamClassifier()
    classifier = st.session_state.classifier
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode",
                                    ["Home", "Upload Data", "Train Model", "Predict", "Visualizations", "About"])
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if app_mode == "Home":
        st.write("""Welcome to the Spam Email Classifier!""")
        if st.button("Load Sample Data"):
            sample_data = {'Category': ['ham', 'spam', 'ham', 'spam'],
                           'Message': ['Hello friend', 'WINNER! Claim your prize', 
                                       'Let’s meet tomorrow', 'URGENT: Account hacked']}
            st.session_state.df = pd.DataFrame(sample_data)
            st.success("Sample data loaded!")
        if st.button("Load email.csv Dataset"):
            try:
                df = pd.read_csv("email.csv")
                st.session_state.df = df
                st.success("email.csv loaded successfully!")
            except Exception as e:
                st.error(f"Could not load dataset: {e}")
    
    elif app_mode == "Upload Data":
        uploaded_file = st.file_uploader("Choose CSV", type="csv")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.dataframe(df.head())
    
    elif app_mode == "Train Model":
        if st.session_state.df is None:
            st.warning("Upload data first!")
            return
        df = st.session_state.df
        model_type = st.selectbox("Select Model", ["naive_bayes", "logistic_regression", "random_forest", "svm"])
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        if st.button("Train Model"):
            X = df['Message'].astype(str)
            y = df['Category']
            X_vectorized, y = classifier.train_model(X, y, model_type)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vectorized, y, test_size=test_size, random_state=42, stratify=y
                )
                classifier.model.fit(X_vectorized, y)
                y_pred = classifier.model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Trained! Accuracy: {acc:.2%}")
            except Exception as e:
                classifier.model.fit(X_vectorized, y)
                st.success("Trained without evaluation due to class imbalance.")
    
    elif app_mode == "Predict":
        if classifier.model is None:
            st.warning("Train a model first!")
            return
        input_method = st.radio("Input Method", ["Single Text", "Batch File"])
        if input_method == "Single Text":
            text_input = st.text_area("Enter message:", height=150)
            if st.button("Classify") and text_input.strip():
                prediction, probability = classifier.predict(text_input)
                if prediction == 'spam':
                    st.markdown('<p class="spam-text">🚫 Classified as SPAM</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="ham-text">✅ Classified as HAM</p>', unsafe_allow_html=True)
                st.metric("Ham Probability", f"{probability[0]:.2%}")
                st.metric("Spam Probability", f"{probability[1]:.2%}")
        else:
            uploaded_file = st.file_uploader("Upload CSV with 'Message'", type="csv")
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                if 'Message' not in batch_df.columns:
                    st.error("CSV must have 'Message' column")
                    return
                results = []
                for msg in batch_df['Message']:
                    prediction, prob = classifier.predict(str(msg))
                    results.append({'Message': msg, 'Prediction': prediction,
                                    'Ham_Prob': prob[0], 'Spam_Prob': prob[1]})
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                st.download_button("Download Results", results_df.to_csv(index=False),
                                   "predictions.csv", "text/csv")
    
    elif app_mode == "Visualizations":
        if st.session_state.df is None:
            st.warning("Upload data first!")
            return
        create_visualizations(st.session_state.df, classifier)
    
    elif app_mode == "About":
        st.write("This app classifies emails as Spam or Ham.")

if __name__ == "__main__":
    main()

