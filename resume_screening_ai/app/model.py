from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

class ResumeClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        self.model = LogisticRegression(max_iter=500)

    def train(self, resumes, labels):
        X = self.vectorizer.fit_transform(resumes)
        self.model.fit(X, labels)

    def predict(self, resume_text, job_desc):
        combined_text = resume_text + " " + job_desc
        X = self.vectorizer.transform([combined_text])
        pred_label = self.model.predict(X)[0]
        pred_prob = max(self.model.predict_proba(X)[0])
        return pred_label, pred_prob

    def save(self, path="models/resume_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load(self, path="models/resume_model.pkl"):
        with open(path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)
