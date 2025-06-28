import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, request, render_template

# --- NLTK Data Download Check (Run once or ensure they are present) ---
# It's recommended to run 'python -m nltk.downloader wordnet omw-1.4 stopwords'
# in your terminal before starting the Flask app for the first time.
# This block attempts to download if not found, but can delay app startup.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# --- Flask App Setup ---
app = Flask(__name__)

# --- 1. Text Preprocessing (from your original script) ---
def preprocess_text(text):
    """
    Cleans and preprocesses text for NLP analysis.
    Steps: lowercasing, removing non-alphanumeric, tokenization, stop word removal, lemmatization.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# --- 2. Keyword/Skill Extraction (from your original script) ---
def extract_keywords_from_jd(job_description_text, keyword_list=None):
    """
    Extracts potential keywords/skills from a job description.
    """
    if not isinstance(job_description_text, str):
        return []

    default_keywords = [
        "python", "java", "sql", "aws", "azure", "gcp", "machine learning",
        "deep learning", "nlp", "data science", "tableau", "power bi",
        "statistics", "pytorch", "tensorflow", "scikit-learn", "spark",
        "hadoop", "ai", "artificial intelligence", "data analysis",
        "communication", "problem solving", "teamwork", "leadership",
        "excel", "api", "rest", "git", "github", "docker", "kubernetes",
        "etl", "data warehousing", "big data", "cloud", "agile",
        "mathematics", "statistics", "algorithms", "modeling", "analytics",
        "transformer", "distilbert", "time series", "feature engineering",
        "lgbm", "lightgbm", "yfinance", "fredapi", "matplotlib", "pandas", "numpy"
    ]
    if keyword_list is None:
        keywords_to_extract = default_keywords
    else:
        keywords_to_extract = keyword_list

    found_keywords = []
    processed_jd = preprocess_text(job_description_text)
    processed_jd_words = set(processed_jd.split())

    for keyword in keywords_to_extract:
        if preprocess_text(keyword) in processed_jd_words or \
           re.search(r'\b' + re.escape(preprocess_text(keyword)) + r'\b', processed_jd):
            found_keywords.append(keyword)
    return found_keywords

def check_resume_for_skills(resume_text, required_skills):
    """
    Checks which required skills are present in a resume.
    Returns a tuple: (matched_skills, missing_skills)
    """
    if not isinstance(resume_text, str):
        return [], required_skills

    processed_resume = preprocess_text(resume_text)
    matched_skills = []
    missing_skills = []

    for skill in required_skills:
        if re.search(r'\b' + re.escape(preprocess_text(skill)) + r'\b', processed_resume):
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)
    return matched_skills, missing_skills

# --- 3. Main Screening Logic (from your original script) ---
def automated_resume_screening(job_description, resumes, jd_keywords_for_scoring=None):
    """
    Screens resumes against a job description and ranks them.
    """
    print("\n--- Starting Automated Resume Screening ---")

    processed_jd = preprocess_text(job_description)
    processed_resumes = {rid: preprocess_text(rtext) for rid, rtext in resumes.items()}

    if jd_keywords_for_scoring is None:
        required_skills_from_jd = extract_keywords_from_jd(job_description)
    else:
        required_skills_from_jd = jd_keywords_for_scoring

    print(f"Identified Key Skills from Job Description: {', '.join(required_skills_from_jd)}")

    all_texts = [processed_jd] + list(processed_resumes.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    jd_vector = tfidf_matrix[0:1]

    screening_results = []

    for i, (resume_id, processed_rtext) in enumerate(processed_resumes.items()):
        resume_vector = tfidf_matrix[i + 1 : i + 2]

        cosine_sim = cosine_similarity(jd_vector, resume_vector)[0][0]

        matched_skills, missing_skills = check_resume_for_skills(
            resumes[resume_id],
            required_skills_from_jd
        )
        keyword_score = len(matched_skills) / len(required_skills_from_jd) if required_skills_from_jd else 0

        total_score = (cosine_sim * 0.7) + (keyword_score * 0.3)
        total_score = round(total_score * 100, 2)

        screening_results.append({
            "Resume ID": resume_id,
            "Total Score (%)": total_score,
            "Cosine Similarity": round(cosine_sim, 2),
            "Matched Skills": matched_skills,
            "Missing Skills": missing_skills
        })

    ranked_results = sorted(screening_results, key=lambda x: x['Total Score (%)'], reverse=True)

    print("--- Screening Complete ---")
    return ranked_results

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main page with the resume screening form."""
    return render_template('index.html', results=None)

@app.route('/screen', methods=['POST'])
def screen_resumes():
    """Handles resume screening requests."""
    job_description = request.form['job_description']
    
    # Get all resume inputs. Flask's request.form.getlist returns a list for multiple inputs with the same name.
    resume_texts = request.form.getlist('resumes')

    # Filter out empty resume fields
    valid_resumes = [res.strip() for res in resume_texts if res.strip()]

    if not job_description.strip():
        return render_template('index.html', error="Please provide a Job Description.")
    if not valid_resumes:
        return render_template('index.html', error="Please provide at least one Resume.")

    # Prepare resumes dictionary for the screening function
    resumes_dict = {f"Resume {i+1}": text for i, text in enumerate(valid_resumes)}

    # Define critical skills (you can customize or automatically extract from JD)
    # For now, we'll re-extract from the job description
    jd_critical_skills = extract_keywords_from_jd(job_description)

    results = automated_resume_screening(
        job_description,
        resumes_dict,
        jd_keywords_for_scoring=jd_critical_skills
    )

    # Convert results to a pandas DataFrame for easier display in template
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results[["Resume ID", "Total Score (%)", "Matched Skills", "Missing Skills", "Cosine Similarity"]]
        df_results = df_results.set_index("Resume ID")
        # Convert DataFrame to HTML table
        results_html = df_results.to_html(classes="table-auto w-full text-left whitespace-no-wrap")
    else:
        results_html = "<p class='text-red-500'>No results to display. Check your inputs.</p>"

    return render_template('index.html', results=results_html)

# --- Run the Flask App ---
if __name__ == '__main__':
    # Create the 'templates' directory if it doesn't exist
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory.")
    
    # Note: For production, use a more robust WSGI server like Gunicorn or uWSGI
    app.run(debug=True) # debug=True enables auto-reloading and better error messages
