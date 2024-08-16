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

# Custom CSS to enhance the app's appearance
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

def build_model(input_dim, output_dim):
    try:
        model = Sequential([
            Dense(32, input_dim=input_dim, activation='relu'),
            Dense(16, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error building the model: {e}")
        return None

def train_model(model, X_train, y_train):
    try:
        history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.2, verbose=0)
        return history
    except Exception as e:
        st.error(f"Error training the model: {e}")
        return None

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

    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose an option", ["Load Data", "Train Model", "Evaluate Model", "Predict Animal Type"])

    if option == "Load Data" or not st.session_state.data_loaded:
        st.header("📊 Load and Prepare Data")
        with st.spinner("Loading data..."):
            data = load_and_prepare_data()
        if data:
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test, st.session_state.label_encoder, st.session_state.scaler, zoo_data = data
            st.session_state.data_loaded = True
            st.success("✅ Data loaded and prepared successfully!")
            st.subheader("Preview of the Zoo Dataset")
            st.dataframe(zoo_data.style.highlight_max(axis=0))
        else:
            st.error("❌ Failed to load data. Please check your data file and try again.")

    elif option == "Train Model":
        st.header("🧠 Train Model")
        if st.session_state.data_loaded:
            if st.button("🚀 Train Model"):
                with st.spinner("Training in progress..."):
                    st.session_state.model = build_model(st.session_state.X_train.shape[1], st.session_state.y_train.shape[1])
                    if st.session_state.model:
                        history = train_model(st.session_state.model, st.session_state.X_train, st.session_state.y_train)
                        if history:
                            st.session_state.model_trained = True
                            st.success("✅ Model trained successfully!")
                            
                            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                            ax[0].plot(history.history['accuracy'], label='Training Accuracy')
                            ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
                            ax[0].set_title('Model Accuracy', fontsize=14)
                            ax[0].set_ylabel('Accuracy', fontsize=12)
                            ax[0].set_xlabel('Epoch', fontsize=12)
                            ax[0].legend(fontsize=10)

                            ax[1].plot(history.history['loss'], label='Training Loss')
                            ax[1].plot(history.history['val_loss'], label='Validation Loss')
                            ax[1].set_title('Model Loss', fontsize=14)
                            ax[1].set_ylabel('Loss', fontsize=12)
                            ax[1].set_xlabel('Epoch', fontsize=12)
                            ax[1].legend(fontsize=10)

                            st.pyplot(fig)
        else:
            st.error("❗ Please load the data first.")

    elif option == "Evaluate Model":
        st.header("📊 Evaluate Model")
        if st.session_state.model_trained:
            if st.button("🔍 Evaluate Model"):
                with st.spinner("Evaluating model..."):
                    try:
                        evaluation_results = evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test)
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
        else:
            st.error("❗ Model is not trained yet. Please train the model first.")

    elif option == "Predict Animal Type":
        st.header("🔮 Predict Animal Type")
        if st.session_state.model_trained:
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
                    prediction = make_prediction(st.session_state.model, st.session_state.scaler, st.session_state.label_encoder, entries)
                if prediction:
                    st.success(f"🎉 The predicted animal is a **{prediction}**.")
        else:
            st.error("❗ Model not trained yet. Please train the model first.")

if __name__ == "__main__":
    main()
