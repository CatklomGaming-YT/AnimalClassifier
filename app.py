import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Animal Type Predictor", page_icon="🐾", layout="wide")

# Custom CSS (unchanged)
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #1c83e1;
    }
    h1 {
        color: #1c83e1;
    }
    h2 {
        color: #0e4e8a;
    }
    h3 {
        color: #3d5a80;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .css-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    try:
        zoo_data = pd.read_csv('zoo.data', header=None)
        X = zoo_data.drop(columns=[0, 17])
        y = zoo_data[17]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)

        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler, zoo_data

    except Exception as e:
        st.error(f"Error loading and preparing data: {e}")
        return None

@st.cache_resource
def build_and_train_model(_X_train, _y_train):
    model = Sequential([
        Dense(32, input_dim=_X_train.shape[1], activation='relu'),
        Dense(16, activation='relu'),
        Dense(_y_train.shape[1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(_X_train, _y_train, epochs=50, batch_size=5, validation_split=0.2, verbose=0)
    return model

def evaluate_model(model, X_test, y_test):
    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        
        return test_loss, test_accuracy, precision, recall, f1
    except Exception as e:
        st.error(f"Error evaluating the model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        st.error(f"Model input shape: {model.input_shape}, output shape: {model.output_shape}")
        return None

def make_prediction(model, scaler, label_encoder, features):
    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        species_mapping = {
            1: 'Mammal', 2: 'Bird', 3: 'Reptile', 4: 'Fish',
            5: 'Amphibian', 6: 'Insect', 7: 'Other'
        }
        predicted_species = species_mapping.get(np.argmax(prediction) + 1, "Unknown")
        return predicted_species
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    st.title("🐾 Animal Type Predictor")
    st.markdown("""
    <p style='font-size: 1.2em;'>
    This application predicts the type of an animal based on various features.
    It uses a neural network model trained on the 
    <a href='https://archive.ics.uci.edu/ml/datasets/Zoo' target='_blank'>Zoo dataset</a> 
    to classify animals into categories such as mammals, birds, reptiles, and more.
    </p>
    """, unsafe_allow_html=True)

    data = load_and_prepare_data()
    if data is not None:
        X_train, X_test, y_train, y_test, label_encoder, scaler, zoo_data = data
        model = build_and_train_model(X_train, y_train)

        st.sidebar.title("Navigation")
        option = st.sidebar.radio("Choose an option", ["Evaluate Model", "Predict Animal Type"])

        if option == "Evaluate Model":
            st.header("📊 Evaluate Model")
            if st.button("🔍 Evaluate Model"):
                with st.spinner("Evaluating model..."):
                    try:
                        evaluation_results = evaluate_model(model, X_test, y_test)
                        if evaluation_results:
                            test_loss, test_accuracy, precision, recall, f1 = evaluation_results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                                st.metric("Test Loss", f"{test_loss:.4f}")
                            with col2:
                                st.metric("Precision", f"{precision:.2%}")
                                st.metric("Recall", f"{recall:.2%}")
                            with col3:
                                st.metric("F1 Score", f"{f1:.2%}")
                        else:
                            st.error("Evaluation failed. Please check the error messages above.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")

        elif option == "Predict Animal Type":
            st.header("🔮 Predict Animal Type")
            st.subheader("Input Animal Features")
            feature_names = [
                "Hair", "Feathers", "Eggs", "Milk", "Airborne",
                "Aquatic", "Predator", "Toothed", "Backbone", "Breathes",
                "Venomous", "Fins", "Legs", "Tail", "Domestic", "Catsize"
            ]
            entries = []
            col1, col2 = st.columns(2)
            for i, feature in enumerate(feature_names):
                with col1 if i % 2 == 0 else col2:
                    if feature == "Legs":
                        value = st.number_input(f"{feature} (Number)", min_value=0, max_value=8, step=1)
                    else:
                        value = st.selectbox(f"{feature}", ["No", "Yes"], index=0)
                    entries.append(1.0 if value == "Yes" else 0.0 if value == "No" else value)

            if st.button("🔍 Predict"):
                with st.spinner("Making prediction..."):
                    prediction = make_prediction(model, scaler, label_encoder, entries)
                if prediction:
                    st.success(f"🎉 The predicted animal is a **{prediction}**.")
    else:
        st.error("Failed to load data. Please check your data file and try again.")

if __name__ == "__main__":
    main()
