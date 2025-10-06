import streamlit as st
from app.extractor import extract_text_from_pdf, extract_text_from_docx
from app.preprocessing import clean_text, extract_keywords
from app.model import ResumeClassifier

# Load trained model
model = ResumeClassifier()
model.load()

st.title("AI-Powered Resume Screening System with Keyword Check")

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

        # AI Prediction
        label, confidence = model.predict(clean_resume, clean_job)

        # Keyword matching
        resume_keywords = extract_keywords(text)
        jd_keywords = extract_keywords(job_desc)
        matched_keywords = resume_keywords.intersection(jd_keywords)
        if len(jd_keywords) > 0:
            keyword_score = len(matched_keywords) / len(jd_keywords)
        else:
            keyword_score = 0.0

        # Final fit status (combine AI confidence and keyword score)
        final_score = (confidence + keyword_score) / 2  # simple average
        threshold = 0.5
        fit_status = "✅ Resume is suitable" if final_score >= threshold else "❌ Resume is NOT suitable"

        # Display results
        st.write(f"**Prediction:** {label}")
        st.write(f"**AI Confidence:** {confidence*100:.2f}%")
        st.write(f"**Keyword Match:** {len(matched_keywords)}/{len(jd_keywords)} ({keyword_score*100:.2f}%)")
        st.write(f"**Final Fit Score:** {final_score*100:.2f}%")
        st.write(f"**Fit Status:** {fit_status}")
