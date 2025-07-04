
import streamlit as st
from transformers import pipeline

# Load the AI model
classifier = pipeline("zero-shot-classification")

# App UI
st.title("💬 Sentiment Classifier with AI")
st.write("Type a sentence and AI will guess the sentiment!")

# User input
user_input = st.text_input("Enter your review or sentence:")

# Candidate labels
labels = ["Positive", "Negative", "Neutral"]

# Predict sentiment
if user_input:
    result = classifier(user_input, candidate_labels=labels)
    top_label = result['labels'][0]
    top_score = result['scores'][0] * 100

    # Display results
    st.subheader("Result:")
    st.write(f"📝 **Review**: {user_input}")
    st.write(f"💡 **Sentiment**: `{top_label}`")
    st.write(f"📊 **Confidence**: `{top_score:.2f}%`")


st.markdown("---")
st.markdown("**Built by Harshavardhan**")
st.markdown("🔗 [LinkedIn](https:/www.linkedin.com/in/harshavardhangh) &nbsp;&nbsp;|&nbsp;&nbsp; "
            "💻 [GitHub](https://github.com/gh-harsha/30DaysOfAI)")
