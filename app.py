import streamlit as st
import joblib



model = joblib.load('intent_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# Page setup
st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🎓", layout="centered")

# Custom CSS styling
st.markdown("👋 Welcome! Type your question below to get career guidance based on Machine Learning.")

# Title and intro
st.title("🎓 Career Guidance Chatbot")
st.markdown("👋 Welcome! Type your question and get career suggestions.")

user_input = st.text_input(
    "Ask a question about a career role:",
    placeholder="e.g. Which role suits someone good at coding?"
)


if st.button('Get Career Suggestion'):
    if user_input:
        
        user_input_vectorized= vectorizer.transform([user_input])
        
        
        prediction = model.predict(user_input_vectorized)
        
        
        st.markdown(f"### ✅ Suggested Career Role: `{prediction[0]}`")
    else:
        st.warning("⚠️ Please enter a question.")
        
# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Sabika Samman | Internship Final Project")
