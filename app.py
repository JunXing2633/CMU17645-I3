import streamlit as st
import mlflow
from config import openai_api_key
import pandas as pd
from gpt_model import analyze_sentiment_gpt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from transformers import pipeline
import time
import matplotlib.pyplot as plt
import seaborn as sns
from bertweet_model import *

st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize MLflow tracking
mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("Sample_Streamlit_App")

def analyze_sentiment_roberta(row):
    sentiment_analysis_roberta_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    analysis_results = sentiment_analysis_roberta_model(row['inputs'])
    max_score = -1
    max_label = ""
    for result in analysis_results:
        if result['score'] > max_score:
            max_score = result['score']
            max_label = result['label']
    return 1 if max_label == "POSITIVE" else 0

def sentiment_analysis_gpt(row):
    analysis_results = analyze_sentiment_gpt(row['inputs'], openai_api_key)
    return 1 if analysis_results == "Positive" else 0


def display_metrics(df):
    # Ensure ground_truth and predictions are integers for metric calculations
    y_true = df['ground_truth'].astype(int)
    y_pred = df['predictions'].astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    
    # Setup layout
    col1, col2= st.columns(2)
    # Display metrics in a table
    
    st.markdown(f"""
        - **Accuracy**: {accuracy * 100:.2f}%
        - **Precision**: {precision * 100:.2f}%
        - **Recall**: {recall * 100:.2f}%
        - **F1 Score**: {f1 * 100:.2f}%
        - **AUC-ROC**: {auc_roc * 100:.2f}%
        """)

    # Calculate and plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    with col1:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic(ROC)')
        plt.legend(loc="lower right")
        st.pyplot()
    
    # Plot confusion matrix
    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)


    return accuracy, precision, recall, f1, auc_roc


# Streamlit UI setup
st.title('Developer-Focused Sentiment Analysis Tool')

with st.sidebar:
    data_points = st.slider('Number of data points to test:', min_value=5, max_value=1000, value=5)
    model_choice = st.selectbox("Select a model for sentiment analysis:", ["GPT", "RoBERTa", "BERTweet"])
    use_mlflow = st.checkbox("Track this run with MLflow")

if 'file_is_valid' not in st.session_state:
    st.session_state['file_is_valid'] = False

file_uploader_container = st.empty()

if not st.session_state['file_is_valid']:
    uploaded_file = file_uploader_container.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        with st.spinner('Loading and validating the file...'):
            try:
                df = pd.read_csv(uploaded_file)
                if set(df.columns) == {'text', 'label'} and df['label'].apply(lambda x: x in [0, 1]).all():
                    st.session_state['file_is_valid'] = True
                    file_uploader_container.empty()
                    st.success("File is valid. You can proceed with sentiment analysis.")
                    
                    st.session_state['df'] = df
                else:
                    st.error("Invalid file format. The file must contain exactly two columns: 'text' (string) and 'label' (int 0 or 1).")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

if st.button("Run Sentiment Analysis"):
    with st.spinner('Analyzing sentiments...'):
        df_subset = st.session_state['df'].sample(n=data_points)
        df_subset.columns = ["inputs", "ground_truth"]
        start_time = time.time()
        if model_choice == "GPT":
            # Ensure to replace "your_openai_api_key" with the actual API key from your configuration
            df_subset['predictions'] = df_subset['inputs'].apply(lambda x: 1 if analyze_sentiment_gpt(x, openai_api_key) == "Positive" else 0)
        elif model_choice == "RoBERTa":
            df_subset['predictions'] = df_subset.apply(analyze_sentiment_roberta, axis=1)
        elif model_choice == "BERTweet":
            df_subset['predictions'] = df_subset.apply(predict_sentiment_bertweet, axis=1)
        
    
    texts_per_second = len(df_subset) / (time.time() - start_time)
    st.info(f"Processed {len(df_subset)} texts. Speed: {texts_per_second:.2f} texts/second")
    accuracy, precision, recall, f1, auc_roc = display_metrics(df_subset)

    if use_mlflow:
        with st.spinner('MLFlow recording...'):
            # MLflow tracking
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("Model", model_choice)

                # Log metrics
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc_roc": auc_roc,
                    "texts_per_second": texts_per_second
                })

                # Log model artifact (for RoBERTa)
                if model_choice == "RoBERTa":
                    mlflow.transformers.log_model(pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english"), "roberta-model")
                
                elif model_choice == "BERTweet":
                    mlflow.transformers.log_model(pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis"), "bertweet-model")
            

                # Log data subset as artifact
                data_subset_path = "data_subset.csv"
                df_subset.to_csv(data_subset_path, index=False)
                mlflow.log_artifact(data_subset_path, "data")

                mlflow.set_tag("Description", "Sentiment analysis with user-uploaded data.")
                st.success("Metrics calculated and logged with MLflow.")

