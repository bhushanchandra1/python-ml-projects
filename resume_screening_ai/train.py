import os
from app.preprocessing import clean_text
from app.model import ResumeClassifier

# Load dummy data
with open("data/resumes.txt", "r") as f:
    resumes = f.readlines()

with open("data/job_desc.txt", "r") as f:
    job_descs = f.readlines()

with open("data/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

texts = [clean_text(r + " " + j) for r, j in zip(resumes, job_descs)]

classifier = ResumeClassifier()
classifier.train(texts, labels)

if not os.path.exists("models"):
    os.makedirs("models")
classifier.save("models/resume_model.pkl")

print("✅ Model trained and saved at models/resume_model.pkl")
