import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    return True

# Load models and vectorizer
@st.cache_resource
def load_models_and_vectorizer():
    """Load all trained models and the TF-IDF vectorizer"""
    model_dir = "notebook/models_notebook"
    
    # Load vectorizer
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer not found at {vectorizer_path}")
        return None, None
    
    vectorizer = joblib.load(vectorizer_path)
    
    
    # Load only hybrid models
    classifier_names = ['random_forest', 'naive_bayes', 'support_vector_machine']
    display_names = ['Random Forest', 'Naive Bayes', 'Support Vector Machine']

    models = {}
    sampling_type = 'hybrid'  # only load this one
    models[sampling_type] = {}

    for clf_name, display_name in zip(classifier_names, display_names):
        model_path = os.path.join(model_dir, f"{clf_name}_{sampling_type}.joblib")
        if os.path.exists(model_path):
            models[sampling_type][display_name] = joblib.load(model_path)

    return models, vectorizer


# Text preprocessing functions
def map_education_level(education):
    if pd.isna(education) or education == "": return 'Unknown'
    level_map = {
        "Bachelor's Degree": 'Bachelor',
        'High School or equivalent': 'High School',
        'Unspecified': 'Unknown',
        "Master's Degree": 'Master',
        'Associate Degree': 'Associate',
        'Certification': 'Certification',
        'Some College Coursework Completed': 'Some College',
        'Professional': 'Professional',
        'Vocational': 'Vocational',
        'Some High School Coursework': 'High School',
        'Doctorate': 'Doctorate',
        'Vocational - HS Diploma': 'High School',
        'Vocational - Degree': 'Associate'
    }
    return level_map.get(education, 'Other')

def map_industry(industry):
    if pd.isna(industry) or industry == "": return 'Unknown'
    level_map = {
        "Marketing and Advertising":"Marketing", "Computer Software":"Technology",
        "Hospital & Health Care":"Healthcare", "Online Media":"Media",
        "Information Technology and Services":"Technology", "Financial Services":"Finance",
        "Management Consulting":"Consulting", "Events Services":"Events",
        "Internet":"Technology", "Facilities Services":"Services",
        "Consumer Electronics":"Consumer Goods", "Telecommunications":"Telecommunications",
        "Consumer Services":"Services", "Construction":"Construction",
        "Oil & Energy":"Energy", "Education Management":"Education",
        "Building Materials":"Construction Materials", "Banking":"Finance",
        "Food & Beverages":"Food", "Food Production":"Food",
        "Health, Wellness and Fitness":"Health", "Insurance":"Finance",
        "E-Learning":"Education", "Cosmetics":"Beauty",
        "Staffing and Recruiting":"Services", "Venture Capital & Private Equity":"Finance",
        "Leisure, Travel & Tourism":"Travel", "Human Resources":"HR",
        "Pharmaceuticals":"Medical", "Farming":"Agriculture",
        "Legal Services":"Law", "Luxury Goods & Jewelry":"Luxury",
        "Machinery":"Manufacturing", "Real Estate":"Property",
        "Mechanical or Industrial Engineering":"Engineering",
        "Public Relations and Communications":"PR", "Consumer Goods":"Consumer Products",
        "Medical Practice":"Medical", "Electrical/Electronic Manufacturing":"Manufacturing",
        "Hospitality":"Services", "Music":"Entertainment",
        "Market Research":"Research", "Automotive":"Automotive",
        "Philanthropy":"Charity", "Utilities":"Services",
        "Primary/Secondary Education":"Education", "Logistics and Supply Chain":"Logistics",
        "Design":"Design", "Gambling & Casinos":"Entertainment",
        "Accounting":"Finance", "Environmental Services":"Environment",
        "Mental Health Care":"Medical", "Investment Management":"Finance",
        "Apparel & Fashion":"Fashion", "Media Production":"Media",
        "Publishing":"Media", "Medical Devices":"Medical",
        "Information Services":"Information", "Retail":"Retail",
        "Sports":"Sports", "Computer Games":"Entertainment",
        "Chemicals":"Chemistry", "Aviation & Aerospace":"Aerospace",
        "Business Supplies and Equipment":"Business", "Program Development":"Development",
        "Computer Networking":"Technology", "Biotechnology":"Biotech",
        "Civic & Social Organization":"Social", "Religious Institutions":"Religion",
        "Warehousing":"Logistics", "Airlines/Aviation":"Aviation",
        "Writing and Editing":"Writing", "Restaurants":"Food",
        "Outsourcing/Offshoring":"Services", "Transportation/Trucking/Railroad":"Transport",
        "Wireless":"Telecommunications", "Investment Banking":"Finance",
        "Nonprofit Organization Management":"Nonprofit", "Libraries":"Education",
        "Computer Hardware":"Technology", "Broadcast Media":"Media",
        "Printing":"Printing", "Graphic Design":"Design",
        "Entertainment":"Entertainment", "Wholesale":"Retail",
        "Research":"Research", "Animation":"Entertainment",
        "Government Administration":"Government", "Capital Markets":"Finance",
        "Computer & Network Security":"Technology", "Semiconductors":"Technology",
        "Security and Investigations":"Security", "Architecture & Planning":"Architecture",
        "Maritime":"Maritime", "Fund-Raising":"Charity",
        "Higher Education":"Education", "Renewables & Environment":"Environment",
        "Motion Pictures and Film":"Entertainment", "Law Practice":"Law",
        "Government Relations":"Government", "Packaging and Containers":"Packaging",
        "Sporting Goods":"Sports", "Mining & Metals":"Mining",
        "Import and Export":"Trade", "International Trade and Development":"Trade",
        "Professional Training & Coaching":"Training", "Textiles":"Textile",
        "Commercial Real Estate":"Property", "Law Enforcement":"Law",
        "Package/Freight Delivery":"Delivery", "Translation and Localization":"Language",
        "Photography":"Photography", "Industrial Automation":"Automation",
        "Wine and Spirits":"Beverage", "Public Safety":"Safety",
        "Civil Engineering":"Engineering", "Military":"Defense",
        "Defense & Space":"Defense", "Veterinary":"Animal",
        "Executive Office":"Management", "Performing Arts":"Arts",
        "Individual & Family Services":"Social", "Public Policy":"Government",
        "Nanotechnology":"Science", "Museums and Institutions":"Culture",
        "Fishery":"Agriculture", "Plastics":"Chemistry",
        "Furniture":"Household", "Shipbuilding":"Maritime",
        "Alternative Dispute Resolution":"Legal", "Ranching":"Agriculture"
    }
    return level_map.get(industry, 'Other')

def map_location(location):
    # --- location mapping ---
    level_map = {
        "AE": "United Arab Emirates", "AL": "Albania", "AM": "Armenia", "AR": "Argentina",
        "AT": "Austria", "AU": "Australia", "BD": "Bangladesh", "BE": "Belgium", "BG": "Bulgaria",
        "BH": "Bahrain", "BR": "Brazil", "BY": "Belarus", "CA": "Canada", "CH": "Switzerland",
        "CL": "Chile", "CM": "Cameroon", "CN": "China", "CO": "Colombia", "CY": "Cyprus",
        "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "EG": "Egypt",
        "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom", "GH": "Ghana",
        "GR": "Greece", "HK": "Hong Kong", "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia",
        "IE": "Ireland", "IL": "Israel", "IN": "India", "IQ": "Iraq", "IS": "Iceland", "IT": "Italy",
        "JM": "Jamaica", "JP": "Japan", "KE": "Kenya", "KH": "Cambodia", "KR": "South Korea",
        "KW": "Kuwait", "KZ": "Kazakhstan", "LK": "Sri Lanka", "LT": "Lithuania", "LU": "Luxembourg",
        "LV": "Latvia", "MA": "Morocco", "MT": "Malta", "MU": "Mauritius", "MX": "Mexico",
        "MY": "Malaysia", "NG": "Nigeria", "NI": "Nicaragua", "NL": "Netherlands", "NO": "Norway",
        "NZ": "New Zealand", "PA": "Panama", "PE": "Peru", "PH": "Philippines", "PK": "Pakistan",
        "PL": "Poland", "PT": "Portugal", "QA": "Qatar", "RO": "Romania", "RS": "Serbia",
        "RU": "Russia", "SA": "Saudi Arabia", "SD": "Sudan", "SE": "Sweden", "SG": "Singapore",
        "SI": "Slovenia", "SK": "Slovakia", "SV": "El Salvador", "TH": "Thailand", "TN": "Tunisia",
        "TR": "Turkey", "TT": "Trinidad and Tobago", "TW": "Taiwan", "UA": "Ukraine", "UG": "Uganda",
        "US": "United States", "Unknown": "Unknown", "VI": "U.S. Virgin Islands", "VN": "Vietnam",
        "ZA": "South Africa", "ZM": "Zambia"
    }
    return level_map.get(location, 'Other')



def clean_text(text):
    if pd.isna(text) or text == "":
        return ""

    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = text.encode("ascii", "ignore").decode("ascii")
    url_pattern = r"https?://\S+|www\.\S+"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    text = re.sub(url_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(email_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)

def preprocess_job_data(job_data):
    """Preprocess job posting data similar to training data"""
    
    for key in job_data:
        if job_data[key] == "" or pd.isna(job_data[key]):
            job_data[key] = "Unknown"
    
    if 'required_education' in job_data:
        job_data['required_education'] = map_education_level(job_data['required_education'])
    
    if 'industry' in job_data:
        job_data['industry'] = map_industry(job_data['industry'])

    if 'location' in job_data:
        loc = str(job_data['location']).strip()
        loc = loc.split(',')[0] if loc else "Unknown"
        job_data['location'] = map_location(loc)
    
    for col in ['has_company_logo', 'has_questions']:
        if col in job_data:
            job_data[col] = 'Yes' if job_data[col] else 'No'
    
    text_fields = [
        'title', 'department', 'location', 'company_profile', 'description',
        'requirements', 'benefits', 'employment_type', 'required_experience',
        'required_education', 'industry', 'function', 'has_company_logo', 'has_questions'
    ]
    
    combined_text = ' '.join([str(job_data.get(field, 'Unknown')) for field in text_fields])
    cleaned_text = clean_text(combined_text)
    
    return cleaned_text

def predict_proba_func(texts):
    """Prediction function that LIME can use"""
    vectorized = st.session_state.vectorizer.transform(texts)
    return st.session_state.selected_model.predict_proba(vectorized)

# Initialize the app
def main():
    st.set_page_config(
        page_title="🕵️ Fake Job Posting Detection",
        page_icon="🕵️",
        layout="wide"
    )
    
    st.title("🕵️ Fake Job Posting Detection System")
    st.markdown("**Detect fraudulent job postings using machine learning with explainable AI**")
    
    download_nltk_data()
    
    if 'models' not in st.session_state or 'vectorizer' not in st.session_state:
        with st.spinner("Loading models and vectorizer..."):
            models, vectorizer = load_models_and_vectorizer()
            if models is None or vectorizer is None:
                st.error("Failed to load models. Please check if model files exist.")
                return
            st.session_state.models = models
            st.session_state.vectorizer = vectorizer
    
    st.sidebar.header("🎛️ Model Configuration")
    
    available_sampling = list(st.session_state.models.keys())
    available_classifiers = []
    
    for sampling_type in available_sampling:
        available_classifiers.extend(list(st.session_state.models[sampling_type].keys()))
    
    available_classifiers = list(set(available_classifiers))
    
    sampling_strategy = "hybrid"
    
    classifier_type = st.sidebar.selectbox(
        "Classifier Type:",
        list(st.session_state.models[sampling_strategy].keys())
    )
    
    st.session_state.selected_model = st.session_state.models[sampling_strategy][classifier_type]
    st.sidebar.success(f"Selected: {classifier_type} ({sampling_strategy})")
    
    tab1, = st.tabs(["🔍 Single Prediction"])
    
    with tab1:
        st.header("🔍 Single Job Posting Analysis")

        # --- START: ADDED SECTION FOR EXAMPLE DATA ---
        
        # Define example data dictionaries
        fraud_data = {
            'title': "Work From Home Data Entry Clerk",
            'company_profile': "Our company is a global leader in providing innovative solutions. We are expanding rapidly and looking for motivated individuals to join our remote team. No experience necessary, we provide full training.",
            'description': "URGENT HIRING! Earn up to $3500 per week. Easy data entry tasks from the comfort of your home. We need your bank account details for direct deposit setup immediately upon application. Limited spots available, apply now! Contact us on Telegram for a faster process.",
            'requirements': "Must have a computer and internet connection. Basic typing skills. No prior experience needed.",
            'benefits': "Flexible hours, weekly pay, work from anywhere. High income potential.",
            'location': "US",
            'department': "",
            'employment_type': "Part-time",
            'required_experience': "Entry level",
            'required_education': "High School or equivalent",
            'industry': "Financial Services",
            'function': "Administrative",
            'has_company_logo': False,
            'has_questions': False
        }

        legit_data = {
            'title': "Senior Software Engineer (Backend)",
            'company_profile': "InnovateTech is a leading SaaS provider in the project management space, helping teams collaborate more effectively. We value diversity and are committed to creating an inclusive environment for all employees. Our mission is to build tools that people love to use.",
            'description': "We are seeking an experienced Senior Backend Engineer to join our core services team. You will be responsible for designing, developing, and maintaining our scalable microservices architecture using Python and AWS. You will collaborate with product managers, designers, and other engineers to deliver high-quality features.",
            'requirements': "5+ years of professional software development experience. Proficiency in Python, Django/Flask. Experience with relational databases (e.g., PostgreSQL). Strong understanding of RESTful APIs and microservices. Experience with cloud platforms like AWS is a plus. BS/MS in Computer Science or a related field.",
            'benefits': "Competitive salary and equity package. Comprehensive health, dental, and vision insurance. 401(k) plan with company match. Generous PTO and paid holidays. Professional development stipend.",
            'location': "US, CA, San Francisco",
            'department': "Engineering",
            'employment_type': "Full-time",
            'required_experience': "Mid-Senior level",
            'required_education': "Bachelor's Degree",
            'industry': "Computer Software",
            'function': "Engineering",
            'has_company_logo': True,
            'has_questions': True
        }

        # Function to load data into session state
        def load_data(data):
            for key, value in data.items():
                st.session_state[key] = value
        
        st.subheader("Load Example Data")
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("Load Example Fraudulent Job", use_container_width=True):
                load_data(fraud_data)
        with col_load2:
            if st.button("Load Example Legitimate Job", use_container_width=True):
                load_data(legit_data)
        
        st.markdown("---") # Visual separator

        # --- END: ADDED SECTION FOR EXAMPLE DATA ---
        
        # --- START: MODIFIED SECTION FOR STATEFUL INPUTS ---
        
        # Initialize session state for form fields if they don't exist
        # This ensures the form remembers its state and can be updated by the buttons
        # if 'title' not in st.session_state:
        #     st.session_state.title = "Data Entry Clerk"
        #     st.session_state.company_profile = "JOIN OUR TEAM! THEN YOU CAN Monthly income exceeds 10,000"
        #     st.session_state.description = "Work from home and earn $5000 per week! Just send us your personal information and we will get you started immediately. No skills required!"
        #     st.session_state.requirements = "Basic computer skills"
        #     st.session_state.benefits = "High pay, work from home. No experience needed"
        #     st.session_state.location = "US, NY, New York"
        #     st.session_state.department = "Customer Service"
        #     st.session_state.employment_type = ""
        #     st.session_state.required_experience = ""
        #     st.session_state.required_education = ""
        #     st.session_state.industry = "Financial Services"
        #     st.session_state.function = "Customer Service..."
        #     st.session_state.has_company_logo = False
        #     st.session_state.has_questions = False
        
        st.subheader("Job Posting Details")
            
        # Use `st.session_state` keys for each widget to make them stateful
        title = st.text_input("Job Title", key='title', placeholder="Enter the job title here")
        company_profile = st.text_area("Company Profile", key='company_profile', placeholder="Enter the company profile here")
        description = st.text_area("Job Description", key='description', placeholder="Enter the job description here")
        requirements = st.text_area("Job Requirements", key='requirements', placeholder="Enter the job requirements here")
        benefits = st.text_area("Job Benefits", key='benefits', placeholder="Enter the job benefits here") 

            
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            location = st.text_input("Location", key='location', placeholder="e.g., US, CA, San Francisco")
            department = st.text_input("Department", key='department', placeholder="e.g., Engineering, Sales, etc.")
            employment_type = st.selectbox("Employment Type", 
                ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"], key='employment_type')
            
        with col1_2:
            required_experience = st.selectbox("Required Experience", 
                ["", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"], key='required_experience')
            required_education = st.selectbox("Required Education", 
                ["", "High School or equivalent", "Bachelor's Degree", "Master's Degree", 
                    "Associate Degree", "Doctorate", "Professional", "Certification"], key='required_education')
            industry = st.text_input("Industry", key='industry', placeholder="e.g., Information Technology and Services")
        
        function = st.text_area("Function", key='function',placeholder="Enter the function here")
        
        col1_3, col1_4 = st.columns(2)
        with col1_3:
            has_company_logo = st.checkbox("Has Company Logo", key='has_company_logo')
        with col1_4:
            has_questions = st.checkbox("Has Screening Questions", key='has_questions')
        
        # --- END: MODIFIED SECTION FOR STATEFUL INPUTS ---
        
        st.subheader("🎯 Prediction Results")
        
        if st.button("🔍 Analyze Job Posting", type="primary"):
            if not st.session_state.title or not st.session_state.description:
                st.error("Please fill in at least Job Title and Description!")
            else:
                # Prepare job data from session state
                job_data = {
                    'title': st.session_state.title,
                    'department': st.session_state.department, 
                    'location': st.session_state.location,
                    'company_profile': st.session_state.company_profile,
                    'description': st.session_state.description,
                    'requirements': st.session_state.requirements,
                    'benefits': st.session_state.benefits,
                    'employment_type': st.session_state.employment_type,
                    'required_experience': st.session_state.required_experience,
                    'required_education': st.session_state.required_education,
                    'industry': st.session_state.industry,
                    'function': st.session_state.function,
                    'has_company_logo': st.session_state.has_company_logo,
                    'has_questions': st.session_state.has_questions
                }
                
                cleaned_text = preprocess_job_data(job_data.copy()) # Pass a copy
                X_vectorized = st.session_state.vectorizer.transform([cleaned_text])
                prediction = st.session_state.selected_model.predict(X_vectorized)[0]
                prediction_proba = st.session_state.selected_model.predict_proba(X_vectorized)[0]
                
                fraud_prob = prediction_proba[1]
                
                if prediction == 1:
                    st.error(f"⚠️ **FRAUDULENT** ({fraud_prob:.1%} confidence)")
                else:
                    st.success(f"✅ **LEGITIMATE** ({(1-fraud_prob):.1%} confidence)")
                
                st.subheader("Confidence Score")
                st.progress(fraud_prob)
                st.write(f"Fraud Probability: {fraud_prob:.3f}")
                
                st.subheader("AI Explanation (LIME)")
                
                with st.spinner("Generating explanation..."):
                    explainer = LimeTextExplainer(class_names=['Legitimate', 'Fraudulent'])
                    explanation = explainer.explain_instance(
                        cleaned_text, 
                        predict_proba_func, 
                        num_features=15,
                        num_samples=1000
                    )
                    
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.markdown("### Prediction probabilities")
                        legitimate_prob = prediction_proba[0]
                        fraudulent_prob = prediction_proba[1]
                        
                        st.markdown("**✅ Legitimate**")
                        st.progress(legitimate_prob, text=f"{legitimate_prob:.2f}")
                        
                        st.markdown("**❌ Fraudulent**")
                        st.progress(fraudulent_prob, text=f"{fraudulent_prob:.2f}")
                        
                        st.markdown("---")
                        
                        exp_list = explanation.as_list()
                        legitimate_features = [(word, abs(weight)) for word, weight in exp_list if weight < 0]
                        fraudulent_features = [(word, weight) for word, weight in exp_list if weight > 0]
                        
                        legitimate_features.sort(key=lambda x: x[1], reverse=True)
                        fraudulent_features.sort(key=lambda x: x[1], reverse=True)
                        
                        tab_leg, tab_fraud = st.tabs(["Legitimate Indicators", "Fraudulent Indicators"])
                        
                        with tab_leg:
                            if legitimate_features:
                                for word, weight in legitimate_features[:10]:
                                    st.markdown(f"**{word}** `{weight:.3f}`")
                            else:
                                st.write("No significant legitimate indicators")
                        
                        with tab_fraud:
                            if fraudulent_features:
                                for word, weight in fraudulent_features[:10]:
                                    st.markdown(f"**{word}** `{weight:.3f}`")
                            else:
                                st.write("No significant fraud indicators")
                    
                    with col_right:
                        st.markdown("### Text with highlighted words")
                        
                        word_weights = dict(exp_list)
                        words = cleaned_text.split()
                        
                        highlighted_html = ""
                        for word in words:
                            if word in word_weights:
                                weight = word_weights[word]
                                if weight > 0:
                                    intensity = min(abs(weight) * 100, 80)
                                    highlighted_html += f'<span style="background-color: rgba(255, 0, 0, {intensity/100}); padding: 2px; margin: 1px; border-radius: 3px;">{word}</span> '
                                elif weight < 0:
                                    intensity = min(abs(weight) * 100, 80)
                                    highlighted_html += f'<span style="background-color: rgba(0, 100, 255, {intensity/100}); padding: 2px; margin: 1px; border-radius: 3px;">{word}</span> '
                                else:
                                    highlighted_html += f"{word} "
                            else:
                                highlighted_html += f"{word} "
                        
                        st.markdown(
                            f'<div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; max-height: 400px; overflow-y: auto; line-height: 1.6;">{highlighted_html}</div>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("""
                        **Legend:**
                        - <span style="background-color: rgba(255, 0, 0, 0.5); padding: 2px; border-radius: 3px;">Red highlight</span>: Indicates fraud
                        - <span style="background-color: rgba(0, 100, 255, 0.5); padding: 2px; border-radius: 3px;">Blue highlight</span>: Indicates legitimate
                        """, unsafe_allow_html=True)
                    
                st.session_state.last_explanation = explanation
                st.session_state.last_job_data = job_data
                st.session_state.last_prediction = {
                    'prediction': prediction,
                    'fraud_probability': fraud_prob,
                    'model_used': f"{classifier_type} ({sampling_strategy})"
                }
    

if __name__ == "__main__":
    main()
    
    st.markdown("---")
    st.markdown(
        "**Note:** This system is designed to assist in identifying potentially fraudulent job postings. "
        "Always use human judgment and additional verification methods for final decisions."
    )