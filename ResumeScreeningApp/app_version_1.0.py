import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import PyPDF2 # For PDF parsing
import docx # For DOCX parsing (python-docx library)

# --- NLTK Data Download Check (Run once or ensure they are present) ---
# It's recommended to run 'python -m nltk.downloader wordnet omw-1.4 stopwords punkt'
# in your terminal before starting the Flask app for the first time.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    print("NLTK stopwords downloaded.")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    print("NLTK punkt tokenizer downloaded.")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("NLTK WordNet downloaded for lemmatization.")
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
    print("NLTK OMW-1.4 downloaded (often needed for WordNet).")


# --- Flask App Setup ---
app = Flask(__name__)
# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set max file size for uploads (e.g., 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'supersecretkey_for_flash_messages' # Needed for flash messages

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload directory: {UPLOAD_FOLDER}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or "" # Ensure non-None for empty pages
        print(f"DEBUG: Successfully extracted text from PDF: {filepath}. Raw length: {len(text)}")
        if not text.strip():
            print(f"DEBUG: PDF '{filepath}' extracted text is empty or only whitespace.")
        return text
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error: Could not read PDF file {filepath}. It might be corrupted, encrypted, or malformed: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while extracting text from PDF {filepath}: {e}")
        return None

def extract_text_from_docx(filepath):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
        print(f"DEBUG: Successfully extracted text from DOCX: {filepath}. Raw length: {len(text)}")
        if not text.strip():
            print(f"DEBUG: DOCX '{filepath}' extracted text is empty or only whitespace.")
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from DOCX {filepath}: {e}")
        return None

# --- 1. Text Preprocessing (from your original script) ---
def preprocess_text(text):
    """
    Cleans and preprocesses text for NLP analysis.
    Steps: lowercasing, removing non-alphanumeric, tokenization, stop word removal, lemmatization.
    """
    if not isinstance(text, str):
        print("DEBUG: preprocess_text received non-string input.")
        return ""

    original_len = len(text)
    # print(f"DEBUG: preprocess_text input (first 50 chars): '{text[:50]}'") # Too verbose for production, but useful for deep debug

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    final_text = ' '.join(words)
    # print(f"DEBUG: preprocess_text output (first 50 chars): '{final_text[:50]}'. Original length: {original_len}, Final length: {len(final_text)}")
    if not final_text.strip():
        print(f"DEBUG: preprocess_text resulted in empty string from original length {original_len}.")
    return final_text

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

    if not required_skills_from_jd:
        print("Warning: No key skills identified from Job Description for scoring. Using fallback general skills.")
        # Fallback: if no specific skills, use all common keywords
        required_skills_from_jd = extract_keywords_from_jd(job_description, keyword_list=[
            "python", "java", "sql", "machine learning", "deep learning", "nlp",
            "data science", "communication", "problem solving"
        ])


    print(f"Identified Key Skills from Job Description: {', '.join(required_skills_from_jd)}")

    # Handle case where no resumes are provided
    if not processed_resumes:
        print("No valid resumes provided for screening (after text extraction).")
        return []

    all_texts = [processed_jd] + list(processed_resumes.values())
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError as e:
        print(f"ERROR: TF-IDF Vectorization failed. This usually means input texts are empty or too short after preprocessing: {e}")
        return []


    jd_vector = tfidf_matrix[0:1]

    screening_results = []

    # Iterate through resumes to calculate scores
    for i, (resume_id, processed_rtext) in enumerate(processed_resumes.items()):
        resume_vector = tfidf_matrix[i + 1 : i + 2]

        cosine_sim = cosine_similarity(jd_vector, resume_vector)[0][0]

        matched_skills, missing_skills = check_resume_for_skills(
            resumes[resume_id],
            required_skills_from_jd
        )
        # Prevent division by zero if no required skills
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
    # RENDER THE NEW HTML FILE HERE!
    return render_template('index__version_1.0.html', results=None, error=None, success_message=None)

@app.route('/screen', methods=['POST'])
def screen_resumes():
    """Handles resume screening requests."""
    # --- NEW DEBUG PRINT HERE ---
    print(f"DEBUG: Received request.form: {request.form}")
    print(f"DEBUG: Received request.files: {request.files}")
    print(f"DEBUG: Files from 'resume_files' input: {request.files.getlist('resume_files')}")
    # --- END NEW DEBUG PRINT ---

    job_description = request.form['job_description']
    
    if not job_description.strip():
        flash("Please provide a Job Description.", 'error')
        print("DEBUG: Job description is empty, redirecting.")
        return redirect(url_for('index'))

    # Handle file uploads
    resumes_dict = {}
    if 'resume_files' in request.files:
        files = request.files.getlist('resume_files')
        print(f"DEBUG: Number of files received from form: {len(files)}")
        for idx, file in enumerate(files):
            print(f"DEBUG: Processing file {idx+1}: {file.filename}, mimetype: {file.mimetype}")
            if file.filename == '':
                print(f"DEBUG: Skipping empty file input (no file selected in field {idx+1}).")
                continue # Skip empty file inputs

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(filepath) # Save the uploaded file temporarily
                    print(f"DEBUG: File '{filename}' saved temporarily to {filepath}")

                    resume_text = None
                    if filename.lower().endswith('.pdf'):
                        resume_text = extract_text_from_pdf(filepath)
                    elif filename.lower().endswith('.docx'):
                        resume_text = extract_text_from_docx(filepath)
                    elif filename.lower().endswith('.txt'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            resume_text = f.read()

                    if resume_text and resume_text.strip(): # Check if extracted text is not empty
                        resumes_dict[filename] = resume_text.strip()
                        print(f"DEBUG: Extracted text from '{filename}'. Length: {len(resume_text.strip())} characters.")
                    else:
                        flash(f"Could not extract meaningful text from '{filename}'. It might be empty, corrupted, or an unsupported format.", 'warning')
                        print(f"DEBUG: Failed to extract meaningful text from '{filename}'. Resume text is empty or None.")
                    
                except Exception as e:
                    flash(f"Error processing file '{filename}': {e}", 'error')
                    print(f"ERROR: Exception while saving/processing '{filename}': {e}")
                finally:
                    # Clean up the temporary file after processing
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"DEBUG: Temporary file '{filepath}' removed.")
            else:
                flash(f"Invalid file type for '{file.filename}'. Only .txt, .pdf, .docx are allowed.", 'error')
                print(f"DEBUG: Invalid file type detected for '{file.filename}'.")
    
    print(f"DEBUG: Final resumes_dict (after processing all files): {list(resumes_dict.keys())}")

    if not resumes_dict:
        flash("Please upload at least one valid resume file (.txt, .pdf, .docx) with content.", 'error')
        print("DEBUG: No valid resumes in resumes_dict, redirecting to index.")
        return redirect(url_for('index'))

    # Define critical skills (you can customize or automatically extract from JD)
    jd_critical_skills = extract_keywords_from_jd(job_description)
    print(f"DEBUG: JD critical skills identified: {jd_critical_skills}")

    results = automated_resume_screening(
        job_description,
        resumes_dict,
        jd_keywords_for_scoring=jd_critical_skills
    )
    print(f"DEBUG: Raw results from screening function: {results}")

    results_html = ""
    if results:
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results[["Resume ID", "Total Score (%)", "Matched Skills", "Missing Skills", "Cosine Similarity"]]
            df_results = df_results.set_index("Resume ID")
            results_html = df_results.to_html(classes="dataframe table-auto w-full text-left whitespace-no-wrap", escape=False) # escape=False for HTML in cells if needed, but not for this output
            flash("Resumes screened successfully!", 'success')
            print("DEBUG: Successfully generated HTML table for results.")
        else:
            results_html = "<p class='text-red-500 font-medium'>No detailed results to display. This might happen if all resumes had no relevant content.</p>"
            flash("Screening completed, but no valid results were generated. Please check inputs.", 'warning')
            print("DEBUG: DataFrame of results was empty after screening, generating placeholder HTML.")
    else:
        results_html = "<p class='text-red-500 font-medium'>No results were generated. Please ensure your Job Description and Resumes have sufficient content.</p>"
        flash("Screening completed, but no valid results were generated. Please check inputs.", 'warning')
        print("DEBUG: Raw results from screening function was empty, generating placeholder HTML.")

    # RENDER THE NEW HTML FILE HERE!
    return render_template('index__version_1.0.html', results=results_html, job_description_value=job_description)

# --- Run the Flask App ---
if __name__ == '__main__':
    # Create the 'templates' directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory.")
    
    app.run(debug=True)
