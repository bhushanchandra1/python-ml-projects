import streamlit as st
from app.extractor import extract_text_from_pdf, extract_text_from_docx
from app.preprocessing import clean_text
from app.model import ResumeClassifier

# Load trained model
model = ResumeClassifier()
model.load()

st.title("AI-Powered Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc = st.text_area("Enter Job Description")

if st.button("Analyze"):
    if uploaded_file and job_desc:
        # Extract text
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_docx(uploaded_file)

        # Clean text
        clean_resume = clean_text(text)
        clean_job = clean_text(job_desc)

        # Predict label + confidence
        label, confidence = model.predict(clean_resume, clean_job)

        # Decide fit based on confidence threshold (e.g., 50%)
        threshold = 0.5
        if confidence >= threshold:
            fit_status = "✅ Resume is suitable for this job description"
        else:
            fit_status = "❌ Resume is NOT suitable for this job description"

        # Display results
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.write(f"**Fit Status:** {fit_status}")
