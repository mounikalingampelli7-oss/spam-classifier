import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vectorizer, accuracy, len(df)


model, vectorizer, accuracy, total = train_model()

st.title("📧 Spam Email Classifier")
st.caption("Type any message below and the model will predict if it's spam or not.")

st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "Naive Bayes")
col2.metric("Training Accuracy", f"{accuracy * 100:.1f}%")
col3.metric("Dataset Size", f"{total:,} messages")

st.markdown("---")

user_input = st.text_area(
    "Enter a message:",
    placeholder="e.g. Congratulations! You've won a free iPhone. Click here to claim.",
    height=120
)

if st.button("🔍 Check Message", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]

        if prediction == 1:
            st.error(f"🚨 **SPAM** — {proba[1]*100:.1f}% confidence")
            st.write("This message looks like spam. Watch out for unsolicited offers, prizes, or urgent calls to action.")
        else:
            st.success(f"✅ **NOT SPAM** — {proba[0]*100:.1f}% confidence")
            st.write("This message looks legitimate.")

st.markdown("---")
st.caption("Built with Python · Scikit-learn · Streamlit · Trained on 5,500+ real SMS messages")