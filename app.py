import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import base64
import io
import shap
from lime.lime_tabular import LimeTabularExplainer
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Background image setup
image_file = 'background.jpg'
with open(image_file, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
loaded_model = joblib.load('traffic_classifier.pkl')

def predict_traffic(test_data):
    if test_data.empty:
        return {}, pd.DataFrame()
    X_test = test_data[['relativetime', 'packetsize', 'packetdirection']]
    predictions = loaded_model.predict(X_test)
    total_predictions = len(predictions)
    class_counts = Counter(predictions)
    class_accuracy_rates = {class_name: count / total_predictions for class_name, count in class_counts.items()}
    return class_accuracy_rates, X_test

def plot_accuracy_rates(accuracy_rates):
    classes = ['Google Search', 'Google Drive', 'Google Music', 'YouTube', 'Google Docs']
    rates = [accuracy_rates.get(i, 0) for i in range(5)]
    fig, ax = plt.subplots()
    ax.bar(classes, rates)
    ax.set_ylabel('Accuracy Rate')
    ax.set_xlabel('Traffic Type')
    ax.set_title('Prediction Accuracy Rates')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def show_shap_explanation(model, X_sample):
    st.subheader("SHAP Explanation (Bar Summary Plot)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap.initjs()
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP failed: {e}")

def show_lime_explanation(model, X_sample):
    st.subheader("LIME Explanation")
    try:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_sample),
            mode='classification',
            feature_names=['relativetime', 'packetsize', 'packetdirection'],
            class_names=['Google Search', 'Google Drive', 'Google Music', 'YouTube', 'Google Docs'],
            discretize_continuous=True
        )
        exp = explainer.explain_instance(
            data_row=X_sample.iloc[0],
            predict_fn=model.predict_proba
        )
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"LIME failed: {e}")

def main():
    st.title('Traffic Classifier App')
    st.sidebar.title('File Selection')

    test_cases = ['GoogleDoc-3.txt', 'GoogleDrive-test1.txt', 'GoogleMusic-8.txt', 'GoogleSearch-7.txt', 'Youtube-20.txt']
    selected_option = st.sidebar.radio('Select a file or upload your own', ('Choose a test file', 'Upload a file'))

    test_data = pd.DataFrame()

    if selected_option == 'Choose a test file':
        selected_file = st.sidebar.selectbox('Select a test file', test_cases)
        test_data = pd.read_csv(selected_file, header=None, sep='\t', names=['timestamp', 'relativetime', 'packetsize', 'packetdirection'])
        st.subheader("Data sample:")
        st.dataframe(test_data.head())

    else:
        uploaded_file = st.sidebar.file_uploader('Upload a .txt file', type='txt')
        if uploaded_file is not None:
            content = uploaded_file.getvalue()
            file_obj = io.StringIO(content.decode('utf-8'))
            test_data = pd.read_csv(file_obj, header=None, sep='\t', names=['timestamp', 'relativetime', 'packetsize', 'packetdirection'])
            st.subheader("Data sample:")
            st.dataframe(test_data.head())

    # Store test_data in session_state so it's retained across reruns
    if not test_data.empty:
        st.session_state["test_data"] = test_data

    if st.button('Run Prediction'):
        accuracy_rates, X_sample = predict_traffic(st.session_state["test_data"])
        st.session_state["X_sample"] = X_sample

        st.write('Prediction Accuracy Rates:')
        max_accuracy = max(accuracy_rates.values())
        for class_no, accuracy_rate in accuracy_rates.items():
            class_name = ['Google Search', 'Google Drive', 'Google Music', 'YouTube', 'Google Docs'][class_no]
            if accuracy_rate == max_accuracy:
                st.markdown(f'<span style="background-color:#009900; padding: 5px">{class_name}: {accuracy_rate:.2f}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="background-color:#990000; padding: 5px">{class_name}: {accuracy_rate:.2f}</span>', unsafe_allow_html=True)

        plot_accuracy_rates(accuracy_rates)

    # Show SHAP/LIME only if prediction has been run
    if "X_sample" in st.session_state:
        if st.button('Show SHAP Explanation'):
            show_shap_explanation(loaded_model, st.session_state["X_sample"])

        if st.button('Show LIME Explanation'):
            show_lime_explanation(loaded_model, st.session_state["X_sample"])

if __name__ == '__main__':
    main()
