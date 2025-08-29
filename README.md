# Fake Job Postings Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://ai-assignment.streamlit.app/)

Predict whether a job posting is **fraudulent or legitimate** using traditional machine learning models and techniques for handling **imbalanced data**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Models Used](#models-used)
- [Deployment](#deployment)
- [How to Run Locally](#how-to-run-locally)
- [Code / Notebook](#code--notebook)
- [Project Structure](#project-structure)
- [Notes](#notes)

---

## Project Overview
This project focuses on detecting **fake job postings** from a dataset of job listings. The workflow includes:

1. Text preprocessing (cleaning and generating `cleaned_text`).  
2. Feature extraction using **TF-IDF**.  
3. Training **machine learning models**: Random Forest, Naive Bayes, and SVM.  
4. Handling **imbalanced data** using methods like undersampling, oversampling, and hybrid approaches.  
5. Deploying the predictive model as a **Streamlit app** for easy interaction.

---

## Models Used
- **Random Forest Classifier**  
- **Naive Bayes Classifier**  
- **Support Vector Machine (SVM)**  

> Each model is evaluated with different strategies to handle imbalanced data for better detection of fraudulent postings.

---

## Deployment
The project is deployed on **Streamlit**:

[🔗 View the App](https://ai-assignment.streamlit.app/)

---

## How to Run Locally
1. Clone the repository:
```bash
git clone https://github.com/your-username/fake-job-detect-nlp.git
cd fake-job-detect-nlp
python -m streamlit run app.py


