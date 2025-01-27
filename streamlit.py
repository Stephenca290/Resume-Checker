import streamlit as st
import PyPDF2
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
import spacy

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

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

# Helper function to perform keyword matching
def extract_keywords(resume_text, keywords):
    matched_keywords = [word for word in keywords if word.lower() in resume_text.lower()]
    return matched_keywords

# Streamlit app
def main():
    st.title("AI-Powered Resume Analysis Tool")

    st.sidebar.header("Upload Your Resume")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF or text file", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = StringIO(uploaded_file.read().decode("utf-8")).getvalue()

        st.subheader("Uploaded Resume Text")
        st.text_area("", resume_text, height=300)

        # Process resume with NLP
        doc = nlp(resume_text)

        # Extract keywords
        matched_keywords = extract_keywords(resume_text, ATS_KEYWORDS)

        # Display results
        st.subheader("ATS Keyword Analysis")
        st.write(f"**Matched Keywords:** {', '.join(matched_keywords)}")

        if matched_keywords:
            st.success(f"Your resume matches {len(matched_keywords)} ATS keywords out of {len(ATS_KEYWORDS)}.")
        else:
            st.warning("No ATS keywords found in your resume. Consider adding relevant skills.")

        # Suggestions
        st.subheader("Suggestions for Improvement")
        st.write("- Ensure your resume includes relevant keywords for the job you're applying for.")
        st.write("- Keep your formatting simple and ATS-friendly.")
        st.write("- Highlight your achievements with measurable outcomes.")

if __name__ == "__main__":
    main()
