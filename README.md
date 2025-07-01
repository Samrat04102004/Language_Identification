# 🌍 Multilingual Language Identifier

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-blue?style=for-the-badge&logo=streamlit)](https://language-identification.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Model-Stacked%20LSTM-red?style=for-the-badge&logo=tensorflow)]()

> 🚀 **A production-ready deep learning solution for real-time detection of 20+ languages via short text inputs.**

---

## 🧠 Overview

**Multilingual Language Identifier** is a powerful natural language processing (NLP) tool designed to detect over 20 languages using a deep learning model. Built with a Stacked LSTM network and deployed with a modern Streamlit interface, this project brings together robust modeling and scalable deployment.

---

## 🛠️ Key Features

- 🌐 **Supports 20+ Languages:** Accurately identifies major world languages like English, Hindi, Spanish, French, Chinese, and more.
- 🤖 **Deep Learning Backbone:** Powered by a Stacked LSTM neural network trained on short and long-form text.
- 📊 **Rich Visual Analytics:** Detailed Exploratory Data Analysis (EDA) and interpretability visuals included in the Jupyter notebook.
- ⚡ **Real-Time Prediction:** Fast and accurate predictions via an interactive Streamlit web interface.

---

## 🚀 Live Demo

👉 **Try it now:** [language-identification.onrender.com](https://language-identification.onrender.com)

---

## 📂 Project Structure

├── main.py # Streamlit app source code

├── language_id_lstm_model.keras # Trained Bidirectional LSTM model

├── tokenizer.joblib # Tokenizer for preprocessing

├── label_encoder.joblib # Label encoder for language classes

├── requirements.txt # Python dependencies

├── runtime.txt # Python runtime version

├── Dockerfile # Container setup for cloud deployment

├── language_id_notebook.ipynb # EDA, preprocessing, and model training

├── README.md # Project overview

---

## 💡 How It Works

1. **Preprocessing** – Text is cleaned and tokenized using a pre-fitted tokenizer.
2. **Model Inference** – Sequences are passed through a trained Bidirectional LSTM network.
3. **Prediction** – The model predicts the most likely language and returns a human-readable label.
4. **Display** – Prediction is shown via Streamlit in real-time.

---

## 🙏 Acknowledgements

- 🤗 [Hugging Face Datasets](https://huggingface.co/datasets)  
- [Streamlit](https://streamlit.io/)  
- [TensorFlow](https://www.tensorflow.org/)  
- [Keras](https://keras.io/)  
- [Scikit-learn](https://scikit-learn.org/)  
- [Render](https://render.com/)

---
