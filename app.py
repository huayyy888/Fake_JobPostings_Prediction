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
        nltk.download('punkt_tab', quiet=True)
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
        "AE": "United Arab Emirates",
        "AL": "Albania",
        "AM": "Armenia",
        "AR": "Argentina",
        "AT": "Austria",
        "AU": "Australia",
        "BD": "Bangladesh",
        "BE": "Belgium",
        "BG": "Bulgaria",
        "BH": "Bahrain",
        "BR": "Brazil",
        "BY": "Belarus",
        "CA": "Canada",
        "CH": "Switzerland",
        "CL": "Chile",
        "CM": "Cameroon",
        "CN": "China",
        "CO": "Colombia",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DE": "Germany",
        "DK": "Denmark",
        "EE": "Estonia",
        "EG": "Egypt",
        "ES": "Spain",
        "FI": "Finland",
        "FR": "France",
        "GB": "United Kingdom",
        "GH": "Ghana",
        "GR": "Greece",
        "HK": "Hong Kong",
        "HR": "Croatia",
        "HU": "Hungary",
        "ID": "Indonesia",
        "IE": "Ireland",
        "IL": "Israel",
        "IN": "India",
        "IQ": "Iraq",
        "IS": "Iceland",
        "IT": "Italy",
        "JM": "Jamaica",
        "JP": "Japan",
        "KE": "Kenya",
        "KH": "Cambodia",
        "KR": "South Korea",
        "KW": "Kuwait",
        "KZ": "Kazakhstan",
        "LK": "Sri Lanka",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "LV": "Latvia",
        "MA": "Morocco",
        "MT": "Malta",
        "MU": "Mauritius",
        "MX": "Mexico",
        "MY": "Malaysia",
        "NG": "Nigeria",
        "NI": "Nicaragua",
        "NL": "Netherlands",
        "NO": "Norway",
        "NZ": "New Zealand",
        "PA": "Panama",
        "PE": "Peru",
        "PH": "Philippines",
        "PK": "Pakistan",
        "PL": "Poland",
        "PT": "Portugal",
        "QA": "Qatar",
        "RO": "Romania",
        "RS": "Serbia",
        "RU": "Russia",
        "SA": "Saudi Arabia",
        "SD": "Sudan",
        "SE": "Sweden",
        "SG": "Singapore",
        "SI": "Slovenia",
        "SK": "Slovakia",
        "SV": "El Salvador",
        "TH": "Thailand",
        "TN": "Tunisia",
        "TR": "Turkey",
        "TT": "Trinidad and Tobago",
        "TW": "Taiwan",
        "UA": "Ukraine",
        "UG": "Uganda",
        "US": "United States",
        "Unknown": "Unknown",
        "VI": "U.S. Virgin Islands",
        "VN": "Vietnam",
        "ZA": "South Africa",
        "ZM": "Zambia"
    }
    return level_map.get(location, 'Other')



def clean_text(text):
    if pd.isna(text) or text == "":
        return ""

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    
    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # Remove URLs and emails
    url_pattern = r"https?://\S+|www\.\S+"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    text = re.sub(url_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(email_pattern, "", text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenize & lowercase (keeping only alphabetic tokens)
    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)

def preprocess_job_data(job_data):
    """Preprocess job posting data similar to training data"""
    
    # Fill missing values
    for key in job_data:
        if job_data[key] == "" or pd.isna(job_data[key]):
            job_data[key] = "Unknown"
    
    # Apply categorical mappings
    if 'required_education' in job_data:
        job_data['required_education'] = map_education_level(job_data['required_education'])
    
    if 'industry' in job_data:
        job_data['industry'] = map_industry(job_data['industry'])

    if 'location' in job_data:
        loc = str(job_data['location']).strip()
        loc = loc.split(',')[0] if loc else "Unknown"
        job_data['location'] = map_location(loc)
    
    # Convert binary flags
    for col in ['has_company_logo', 'has_questions']:
        if col in job_data:
            job_data[col] = 'Yes' if job_data[col] else 'No'
    
    # Combine all text fields
    text_fields = [
        'title', 'department', 'location', 'company_profile', 'description',
        'requirements', 'benefits', 'employment_type', 'required_experience',
        'required_education', 'industry', 'function', 'has_company_logo', 'has_questions'
    ]
    
    combined_text = ' '.join([str(job_data.get(field, 'Unknown')) for field in text_fields])
    
    # Clean the combined text
    cleaned_text = clean_text(combined_text)
    
    return cleaned_text

# Prediction function for LIME
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
    
    # Download NLTK data
    download_nltk_data()
    
    # Load models
    if 'models' not in st.session_state or 'vectorizer' not in st.session_state:
        with st.spinner("Loading models and vectorizer..."):
            models, vectorizer = load_models_and_vectorizer()
            if models is None or vectorizer is None:
                st.error("Failed to load models. Please check if model files exist.")
                return
            st.session_state.models = models
            st.session_state.vectorizer = vectorizer
    
    # Sidebar for model selection
    st.sidebar.header("🎛️ Model Configuration")
    
    # Get available sampling strategies and classifiers
    available_sampling = list(st.session_state.models.keys())
    available_classifiers = []
    
    for sampling_type in available_sampling:
        available_classifiers.extend(list(st.session_state.models[sampling_type].keys()))
    
    available_classifiers = list(set(available_classifiers))
    
    # Model selection
    sampling_strategy = "hybrid"
    
    classifier_type = st.sidebar.selectbox(
        "Classifier Type:",
        list(st.session_state.models[sampling_strategy].keys())
    )
    
    # Load selected model
    st.session_state.selected_model = st.session_state.models[sampling_strategy][classifier_type]
    st.sidebar.success(f"Selected: {classifier_type} ({sampling_strategy})")
    
    # Main content tabs
    tab1, = st.tabs(["🔍 Single Prediction"])
    
    with tab1:
        st.header("🔍 Single Job Posting Analysis")
        
        #col1, col2 = st.columns([2, 1])
        
        #with col1:
            # Input form
        st.subheader("Job Posting Details")
            
        title = st.text_input("Job Title", "Data Entry Clerk")
        company_profile = st.text_area("Company Profile","JOIN OUR TEAM! THEN YOU CAN Monthly income exceeds 10,000")
        description = st.text_area("Job Description*", "Work from home and earn $5000 per week! Just send us your personal information and we will get you started immediately. No skills required!", height=150)
        requirements = st.text_area("Requirements", "Basic computer skills")
        benefits = st.text_area("Benefits", "High pay, work from home. No experience needed")
            
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            location = st.text_input("Location", "US, NY, New York")
            department = st.text_input("Department", "Customer Service")
            employment_type = st.selectbox("Employment Type", 
                ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"])
            
        with col1_2:
            required_experience = st.selectbox("Required Experience", 
                ["", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"])
            required_education = st.selectbox("Required Education", 
                ["", "High School or equivalent", "Bachelor's Degree", "Master's Degree", 
                    "Associate Degree", "Doctorate", "Professional", "Certification"])
            industry = st.text_input("Industry", "Financial Services")
        
        function = st.text_area("Function", "Customer Service...")
        
        col1_3, col1_4 = st.columns(2)
        with col1_3:
            has_company_logo = st.checkbox("Has Company Logo")
        with col1_4:
            has_questions = st.checkbox("Has Screening Questions")
        
        st.subheader("🎯 Prediction Results")
        
        if st.button("🔍 Analyze Job Posting", type="primary"):
            if not title or not description:
                st.error("Please fill in at least Job Title and Description!")
            else:
                # Prepare job data
                job_data = {
                    'title': title,
                    'department': department, 
                    'location': location,
                    'company_profile': company_profile,
                    'description': description,
                    'requirements': requirements,
                    'benefits': benefits,
                    'employment_type': employment_type,
                    'required_experience': required_experience,
                    'required_education': required_education,
                    'industry': industry,
                    'function': function,
                    'has_company_logo': has_company_logo,
                    'has_questions': has_questions
                }
                
                # Preprocess the data
                cleaned_text = preprocess_job_data(job_data)
                
                # Vectorize
                X_vectorized = st.session_state.vectorizer.transform([cleaned_text])
                
                # Make prediction
                prediction = st.session_state.selected_model.predict(X_vectorized)[0]
                prediction_proba = st.session_state.selected_model.predict_proba(X_vectorized)[0]
                
                # Display results
                fraud_prob = prediction_proba[1]
                
                if prediction == 1:
                    st.error(f"⚠️ **FRAUDULENT** ({fraud_prob:.1%} confidence)")
                else:
                    st.success(f"✅ **LEGITIMATE** ({(1-fraud_prob):.1%} confidence)")
                
                # Probability bar
                st.subheader("Confidence Score")
                st.progress(fraud_prob)
                st.write(f"Fraud Probability: {fraud_prob:.3f}")
                
                # LIME Explanation
                st.subheader("AI Explanation (LIME)")
                
                with st.spinner("Generating explanation..."):
                    # Create LIME explainer
                    explainer = LimeTextExplainer(class_names=['Legitimate', 'Fraudulent'])
                    
                    # Generate explanation
                    explanation = explainer.explain_instance(
                        cleaned_text, 
                        predict_proba_func, 
                        num_features=15,
                        num_samples=1000
                    )
                    
                    # Create layout similar to your image
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        # Prediction probabilities section
                        st.markdown("### Prediction probabilities")
                        
                        # Create probability bars
                        legitimate_prob = prediction_proba[0]
                        fraudulent_prob = prediction_proba[1]
                        
                        # Legitimate bar
                        st.markdown("**✅ Legitimate**")
                        st.progress(legitimate_prob, text=f"{legitimate_prob:.2f}")
                        
                        # Fraudulent bar  
                        st.markdown("**❌ Fraudulent**")
                        st.progress(fraudulent_prob, text=f"{fraudulent_prob:.2f}")
                        
                        st.markdown("---")
                        
                        # Feature importance list
                        exp_list = explanation.as_list()
                        
                        # Separate positive and negative weights
                        legitimate_features = [(word, abs(weight)) for word, weight in exp_list if weight < 0]
                        fraudulent_features = [(word, weight) for word, weight in exp_list if weight > 0]
                        
                        # Sort by weight magnitude
                        legitimate_features.sort(key=lambda x: x[1], reverse=True)
                        fraudulent_features.sort(key=lambda x: x[1], reverse=True)
                        
                        # Display tabs for Legitimate and Fraudulent
                        tab_leg, tab_fraud = st.tabs(["Legitimate", "Fraudulent"])
                        
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
                        # Text with highlighted words section
                        st.markdown("### Text with highlighted words")
                        
                        # Get the original words and their importance
                        word_weights = dict(exp_list)
                        
                        # Split cleaned text into words for highlighting
                        words = cleaned_text.split()
                        
                        # Create highlighted text HTML
                        highlighted_html = ""
                        for word in words:
                            if word in word_weights:
                                weight = word_weights[word]
                                if weight > 0:  # Fraudulent indicator
                                    # Red background for fraud indicators
                                    intensity = min(abs(weight) * 100, 80)  # Cap intensity
                                    highlighted_html += f'<span style="background-color: rgba(255, 0, 0, {intensity/100}); padding: 2px; margin: 1px; border-radius: 3px;">{word}</span> '
                                elif weight < 0:  # Legitimate indicator  
                                    # Blue background for legitimate indicators
                                    intensity = min(abs(weight) * 100, 80)
                                    highlighted_html += f'<span style="background-color: rgba(0, 100, 255, {intensity/100}); padding: 2px; margin: 1px; border-radius: 3px;">{word}</span> '
                                else:
                                    highlighted_html += f"{word} "
                            else:
                                highlighted_html += f"{word} "
                        
                        # Display highlighted text
                        st.markdown(
                            f'<div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; max-height: 400px; overflow-y: auto; line-height: 1.6;">{highlighted_html}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Legend
                        st.markdown("""
                        **Legend:**
                        - <span style="background-color: rgba(255, 0, 0, 0.5); padding: 2px; border-radius: 3px;">Red highlight</span>: Indicates fraud
                        - <span style="background-color: rgba(0, 100, 255, 0.5); padding: 2px; border-radius: 3px;">Blue highlight</span>: Indicates legitimate
                        """, unsafe_allow_html=True)
                    
                # Store explanation in session state for download
                st.session_state.last_explanation = explanation
                st.session_state.last_job_data = job_data
                st.session_state.last_prediction = {
                    'prediction': prediction,
                    'fraud_probability': fraud_prob,
                    'model_used': f"{classifier_type} ({sampling_strategy})"
                }
    

if __name__ == "__main__":
    
    # Run main app
    main()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This system is designed to assist in identifying potentially fraudulent job postings. "
        "Always use human judgment and additional verification methods for final decisions."
    )