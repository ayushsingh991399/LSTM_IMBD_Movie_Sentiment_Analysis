# app.py
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 30px;
}
.card {
    background-color: #f9fafb;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.result-positive {
    color: #16a34a;
    font-size: 26px;
    font-weight: 700;
}
.result-negative {
    color: #dc2626;
    font-size: 26px;
    font-weight: 700;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: #6b7280;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🎬 IMDB Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Analyze movie reviews using an LSTM deep learning model</div>',
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    review_text = st.text_area(
        "✍️ Enter Movie Review",
        height=180,
        placeholder="Type or paste a movie review here..."
    )
    predict_btn = st.button("🔍 Predict Sentiment", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


if predict_btn:
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a movie review before predicting.")
    else:
        seq = tokenizer.texts_to_sequences([review_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        prob = float(model.predict(padded)[0][0])
        sentiment = "Positive" if prob >= 0.5 else "Negative"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if sentiment == "Positive":
            st.markdown(
                f'<div class="result-positive">😊 Positive Review</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-negative">😞 Negative Review</div>',
                unsafe_allow_html=True
            )

        st.write(f"**Confidence Score:** {round(prob * 100, 2)}%")
        st.progress(min(max(prob, 0.0), 1.0))

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
  Built By ❤️ Ayush Singh |
  <a href="https://ayush-singh-09.vercel.app/" target="_blank">
    Portfolio
  </a>
</div>
""", unsafe_allow_html=True)
