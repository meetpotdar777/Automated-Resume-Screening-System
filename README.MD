📄 AI-Powered Resume Screening System

This project develops an intelligent web application that automates the initial screening of resumes against a specific job description. It leverages Natural Language Processing (NLP) techniques, machine learning, and a Flask web interface to efficiently rank candidates based on their relevance.

✨ Features

Web-based User Interface: Access the system easily via your web browser with a clean, interactive form.

Job Description Input: Paste or type the job description for the role you're hiring for.

Multiple Resume Uploads: Upload one or more resumes simultaneously in .txt, .pdf, or .docx formats.

Text Extraction: Automatically extracts text content from uploaded PDF and DOCX files.

Advanced Text Preprocessing: Cleans and prepares both job description and resume text using techniques like lowercasing, punctuation removal, stop word filtering, and lemmatization.

Keyword & Skill Matching: Identifies key skills in the job description and checks for their presence in resumes.

Semantic Similarity Scoring: Utilizes TF-IDF vectorization and Cosine Similarity to measure the overall content relevance between the job description and each resume.

Composite Scoring & Ranking: Combines keyword matching and semantic similarity into a total relevance score, then ranks candidates from highest to lowest.

Clear Results Display: Presents screening results in an easy-to-read table showing resume ID, total score, matched skills, and missing skills.

User Feedback: Provides flash messages for success, warnings, or errors directly on the web page.

Loading Indicator: Shows a visual cue during processing for better user experience.

🛠️ Installation

To set up and run this application, follow these steps:

Project Setup:

Create a main project folder (e.g., Automated Resume Screening System).

Inside this folder, save the Python application as app_version_1.0.py.

Create a subfolder named templates within your main project folder.

Inside the templates folder, save the HTML frontend code as index__version_1.0.html.

Your directory structure should look like this:

Automated Resume Screening System/
├── app_version_1.0.py
└── templates/
    └── index__version_1.0.html

Python Environment:

It's highly recommended to use a virtual environment. Open your terminal/command prompt and navigate to your project directory (Automated Resume Screening System).

Create a virtual environment (if you don't have one):

python -m venv venv_resume_screener

Activate the virtual environment:

On Windows:

.\venv_resume_screener\Scripts\activate

On macOS/Linux:

source venv_resume_screener/bin/activate

You should see (venv_resume_screener) at the beginning of your prompt after activation.

Install Dependencies:

With your virtual environment activated, install the necessary Python libraries:

python -m pip install Flask pandas scikit-learn nltk PyPDF2 python-docx tabulate

Download NLTK Data:

The application requires specific NLTK data packages. Run these commands while your virtual environment is active:

python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4

(These are included in app_version_1.0.py to attempt download if missing, but running them manually first ensures a smoother startup).

🚀 Usage

Once everything is set up, you can run the application:

Start the Flask Application:

In your activated terminal, navigate to your project directory (Automated Resume Screening System).

Run the application:

python app_version_1.0.py

You should see output indicating the server is running, usually on http://127.0.0.1:5000.

Access in Browser:

Open your web browser and navigate to the address: http://127.0.0.1:5000

Important: If you previously opened index.html directly from your file system, perform a hard refresh (Ctrl + Shift + R on Windows/Linux, Cmd + Shift + R on Mac) to ensure the latest HTML is loaded.

Perform Resume Screening:

Job Description: Paste or type the job description into the "Job Description" text area.

Upload Resumes:

Prepare your resumes as .txt, .pdf, or .docx files. (For .txt files, use a plain text editor like Notepad; for PDFs, ensure they contain selectable text, not scanned images).

Click the "Choose Files" / "Browse" button under "Upload Resumes (PDF, DOCX, TXT):".

In the file selection dialog, select one or more resume files (hold Ctrl to select multiple non-consecutive files, or Shift for a range).

After selecting, you should see the names of the chosen files appear on the webpage itself, providing visual confirmation.

Click the "Screen Resumes" button.

View Results:

The page will show a loading indicator.

Once processing is complete, a table will appear with the ranked resumes, their scores, matched skills, and missing skills.

You'll also see flash messages at the top for success, warnings, or errors.

📂 File Structure

Automated Resume Screening System/
├── app_version_1.0.py       # Main Flask application with screening logic
├── templates/               # Folder for HTML templates
│   └── index__version_1.0.html # Frontend UI for the web application
└── uploads/                 # Temporary folder for uploaded resume files (created automatically)

🚧 Troubleshooting Tips

Fatal error in launcher or No such file or directory during pip install or python app_version_1.0.py:

This often means your virtual environment is not correctly activated, or your system's pip/python is pointing to a different (possibly broken) Python installation.

Solution: Ensure you are always running commands like python -m pip install ... or python app_version_1.0.py after activating your virtual environment (.\venv_resume_screener\Scripts\activate) and from the correct project directory.

LookupError: Resource wordnet not found (or other NLTK data):

Solution: Run python -m nltk.downloader wordnet omw-1.4 stopwords punkt in your activated virtual environment.

"Please upload at least one valid resume file..." message (or no results):

This indicates Flask did not receive any usable file content.

Check 1: Browser File Selection: Did you click the "Choose Files" button and select actual .txt, .pdf, or .docx files from your computer? Did you see the filenames appear on the webpage under "Selected Files:" before submitting? If not, the browser isn't sending them.

Check 2: File Content: Are your .pdf files scanned images? PyPDF2 cannot extract text from image-based PDFs. Ensure PDFs contain selectable text. Are your .txt or .docx files empty or corrupted?

Check 3: Terminal Debug Messages: After submission, observe your terminal for Flask's debug output (lines starting with DEBUG: or ERROR:). Look for:

DEBUG: Received request.files: - Does this show information about your uploaded files? If it's ImmutableMultiDict([]), no files were sent.

DEBUG: Successfully extracted text from ... Raw length: XXX - Is XXX a non-zero number?

DEBUG: Extracted text from '...' Length: YYY characters. - Is YYY a non-zero number?

Any ERROR messages during file processing or TF-IDF Vectorization failed.

💡 Future Enhancements

Advanced Resume Parsing: Integrate more sophisticated libraries or APIs for parsing different resume formats, handling tables, sections, and metadata extraction (e.g., years of experience, contact info).

Semantic Skill Matching: Utilize pre-trained Transformer models (like BERT, RoBERTa) fine-tuned for Named Entity Recognition (NER) to precisely extract skills and job titles, enabling more nuanced matching.

Weighted Requirements: Allow users to assign different importance levels (e.g., "must-have" vs. "nice-to-have") to skills/keywords in the job description.

User Authentication & Data Storage: Implement user login, and use a database (e.g., SQLite, PostgreSQL) to store job descriptions, resumes, and screening results for historical tracking and multi-user access.



Visualizations: Add charts and graphs to visualize skill overlap, score distribution, or other insights.

Deployment: Deploy the Flask application to a cloud platform (e.g., Heroku, AWS, Azure, Google Cloud) for public access.

Feedback Loop: Implement a way for users to provide feedback on screening results to refine the model over time.

