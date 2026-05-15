# 📧 Spam Message Classifier

A machine learning web app that detects whether a message is spam or not in real-time.

## 🔗 Live Demo
[Click here to try the app](https://mounika-spam-classifier.streamlit.app)

## 🛠️ Tech Stack
- Python
- Scikit-learn (Naive Bayes, CountVectorizer)
- Pandas
- Streamlit

## 📊 Dataset
SMS Spam Collection Dataset — 5,574 labeled messages from Kaggle

## ⚙️ How it works
1. Messages are converted to numerical vectors using CountVectorizer
2. A Multinomial Naive Bayes model is trained on 80% of the data
3. Achieves ~98% accuracy on the test set

## 💻 Run Locally
pip install -r requirements.txt
streamlit run app.py