import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import re

# Predefined ATS keywords
ATS_KEYWORDS = [
    "data analysis", "machine learning", "python", "deep learning", "SQL",
    "cloud computing", "NLP", "project management", "team collaboration", "problem solving"
]

# Helper function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Helper function to preprocess text
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

# Helper function to calculate TF-IDF and match keywords
def match_keywords_with_tfidf(resume_text, keywords):
    # Combine resume text and keywords into a single corpus
    corpus = [resume_text] + keywords

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Extract scores for resume text (first row in the TF-IDF matrix)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix[0].toarray()[0]

    # Match with predefined keywords
    matched_keywords = []
    for keyword in keywords:
        keyword_terms = keyword.split()
        for term in keyword_terms:
            if term in feature_names:
                idx = feature_names.tolist().index(term)
                if tfidf_scores[idx] > 0:
                    matched_keywords.append(keyword)

    return list(set(matched_keywords))

# Streamlit app
def main():
    st.title("Resume Analysis with TF-IDF")

    st.sidebar.header("Upload Your Resume")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(uploaded_file)

        # Preprocess resume text
        cleaned_text = preprocess_text(resume_text)

        # Perform keyword matching
        matched_keywords = match_keywords_with_tfidf(cleaned_text, ATS_KEYWORDS)

        # Display results
        st.subheader("Uploaded Resume Text")
        st.text_area("", resume_text, height=300)

        st.subheader("Matched ATS Keywords")
        if matched_keywords:
            st.success(f"Matched Keywords: {', '.join(matched_keywords)}")
        else:
            st.warning("No ATS keywords matched. Consider revising your resume.")

        # Display suggestions
        st.subheader("Suggestions for Improvement")
        st.write("- Add relevant keywords based on job descriptions.")
        st.write("- Focus on measurable achievements.")
        st.write("- Simplify formatting to ensure ATS compatibility.")

if __name__ == "__main__":
    main()
