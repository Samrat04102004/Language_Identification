import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

lang_code_to_name = {
    'ar': 'Arabic', 'bg': 'Bulgarian', 'de': 'German', 'el': 'Greek', 'en': 'English',
    'es': 'Spanish', 'fr': 'French', 'hi': 'Hindi', 'it': 'Italian', 'ja': 'Japanese',
    'nl': 'Dutch', 'pl': 'Polish', 'pt': 'Portuguese', 'ru': 'Russian', 'sw': 'Swahili',
    'th': 'Thai', 'tr': 'Turkish', 'ur': 'Urdu', 'vi': 'Vietnamese', 'zh': 'Chinese'
}



st.set_page_config(page_title="Language Identifier", page_icon="🌍", layout="centered")
st.title("🌍 Language Identification App")
st.markdown(
    """
    <style>
    .big-font {font-size:22px !important;}
    .stButton>button {background-color: #0099ff; color: white;}
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="big-font">Enter a sentence to detect its language instantly!</div>', unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("language_id_lstm_model.keras")
    tokenizer = joblib.load("tokenizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()


st.subheader("Try it out:")
user_input = st.text_area("Type or paste a sentence here:", height=100)

if st.button("Predict Language"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:

        seq = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=75, padding='post', truncating='post')

        pred = model.predict(padded)
        pred_code = label_encoder.inverse_transform([np.argmax(pred)])[0]
        pred_name = lang_code_to_name.get(pred_code, pred_code)
        st.success(f"**Predicted Language:** {pred_name}")


        probs = pred[0]
        top3_idx = probs.argsort()[-3:][::-1]
        st.markdown("#### Top 3 Predictions:")
        for idx in top3_idx:
            code = label_encoder.inverse_transform([idx])[0]
            name = lang_code_to_name.get(code, code)
            st.write(f"{name}: {probs[idx] * 100:.2f}%")


with st.expander("ℹ️ About this app"):
    st.write("""
    - **Model:** Stacked LSTM trained on 20 languages.
    - **Input:** Any short-medium sentence preferably up to 75 tokens.
    - **Output:** Instant language prediction and top 3 probabilities.
    - **Built with:** TensorFlow, Streamlit, and joblib.
    """)

