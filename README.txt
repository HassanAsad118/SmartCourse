===========================================================================
        SMARTCOURSE - AI-POWERED COURSE RECOMMENDATION SYSTEM
===========================================================================

---------------------------------------------------------------------------
PROJECT OVERVIEW
---------------------------------------------------------------------------
SmartCourse is a full-stack AI-based course recommendation system that 
suggests relevant online courses based on user queries. It uses two 
different machine learning approaches - TF-IDF and Neural Embeddings - 
and allows users to compare their results side-by-side.

The system also provides features like saving courses, viewing 
recommendation history, and directly accessing course links.


---------------------------------------------------------------------------
AUTHOR INFORMATION
---------------------------------------------------------------------------
Name:           Hassan Asad  
Roll Number:    BC220417166 
Program:        BS Software Engineering
University:     Virtual University of Pakistan  
Project:        Final Year Project  
Title:          SmartCourse - AI-Powered Course Recommendation System  


---------------------------------------------------------------------------
FEATURES
---------------------------------------------------------------------------
* Search courses using natural language queries
* Two recommendation models:
  - TF-IDF (keyword-based)
  - Neural (semantic similarity using SentenceTransformer)
* "Both Mode" - compare TF-IDF and Neural results side-by-side
* Save and manage favorite courses
* View search history
* "View Course" button to open course link
* Evaluation metrics:
  - Precision@K
  - Recall@K
  - Hit Rate


---------------------------------------------------------------------------
MACHINE LEARNING MODELS
---------------------------------------------------------------------------
1. TF-IDF MODEL
   - Uses Term Frequency-Inverse Document Frequency
   - Captures keyword-based similarity
   - Faster and interpretable

2. NEURAL MODEL
   - Uses SentenceTransformer (all-MiniLM-L6-v2)
   - Generates semantic embeddings
   - Captures meaning beyond keywords


---------------------------------------------------------------------------
EVALUATION METRICS
---------------------------------------------------------------------------
The system evaluates both models using:

- Precision@10: How many recommended results are relevant
- Recall@10:    How much of total relevant data is captured
- Hit Rate@10:  Whether at least one relevant result is found


---------------------------------------------------------------------------
PROJECT STRUCTURE
---------------------------------------------------------------------------
SmartCourse/
|
|-- app.py
|-- config.py
|-- train_models.py
|-- requirements.txt
|-- README.txt
|-- database.db
|
|-- utils/
|   |-- neural_model.py
|   |-- preprocessing.py
|   |-- tfidf_model.py
|   |-- evaluation.py
|
|-- models/
|   |-- tfidf_model.pkl
|   |-- neural_embeddings.pkl
|
|-- templates/
|   |-- base.html
|   |-- home.html
|   |-- recommend.html
|   |-- dashboard.html
|   |-- about.html
|
|-- static/
|   |-- css/
|   |   |-- style.css
|   |-- js/
|   |   |-- main.js
|   |-- images/
|
|-- data/
    |-- Courses_dataset.csv


---------------------------------------------------------------------------
TECH STACK
---------------------------------------------------------------------------
- Backend:    Flask (Python)
- Frontend:   HTML, CSS, JavaScript, Bootstrap 5
- ML/NLP:     scikit-learn (TF-IDF), sentence-transformers, spaCy
- Database:   SQLite


---------------------------------------------------------------------------
INSTALLATION AND SETUP
---------------------------------------------------------------------------
1. Create Virtual Environment:
   python -m venv venv

2. Activate Virtual Environment:
   venv\Scripts\activate    (Windows)

3. Install Dependencies:
   pip install -r requirements.txt

4. Install spaCy Model (Important):
   python -m spacy download en_core_web_sm

5. Run Application:
   python app.py

6. Access in Browser:
   http://127.0.0.1:5000/


---------------------------------------------------------------------------
HOW IT WORKS
---------------------------------------------------------------------------
1. User enters a query.
2. Text is cleaned and preprocessed.
3. Query is passed to both the TF-IDF and Neural models.
4. Cosine similarity is computed against the dataset.
5. Top-K results are returned based on the highest scores.
6. Results are displayed in the UI (single or comparison mode).


---------------------------------------------------------------------------
DATABASE & EVALUATION
---------------------------------------------------------------------------
DATABASE:
- SQLite database (database.db)
- Stores User search history and Saved courses.

EVALUATION:
- Run model comparison using the following command:
  python -m utils.evaluation


---------------------------------------------------------------------------
IMPORTANT NOTES
---------------------------------------------------------------------------
- Make sure the spaCy model is installed before running the app.
- Models are pre-trained and saved in the /models directory.
- The application runs locally using the Flask development server.

===========================================================================
                      END OF README FILE
===========================================================================