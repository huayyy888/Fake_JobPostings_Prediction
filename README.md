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

---

## Project Overview
This project focuses on detecting **fake job postings** from a dataset of job listings. The workflow includes:

1. Text preprocessing (cleaning and generating `cleaned_text`).  
2. Feature extraction using **TF-IDF**.  
3. Training **machine learning models**: Random Forest, Naive Bayes, and SVM.  
4. Handling **imbalanced data** using methods like undersampling, oversampling, and hybrid approaches.  
5. Deploying the predictive model as a **Streamlit app** for easy interaction.

> The Coding will put the **notebook folder** that contains ipynb notebook and model joblib.<br>
> For the deployment streamlit webapp is **app.py**

---

## Models Used
- **Random Forest Classifier**  
- **Naive Bayes Classifier**  
- **Support Vector Machine (SVM)**  

> Each model is evaluated with different strategies to handle imbalanced data for better detection of fraudulent postings.

---

## Deployment
The project is deployed on **Streamlit**:
[🔗 View the App](https://fakejobpostingsprediction.streamlit.app/)

---

## How to Run Locally

Run the following commands in your terminal:

```bash
git clone https://github.com/huayyy888/Fake_JobPostings_Prediction.git
cd Fake_JobPostings_Prediction
python -m streamlit run app.py
```
## Future Work 
Explore BERT or other neural network models for improved text understanding and prediction accuracy.
Implement advanced ensemble methods for better performance on imbalanced datasets.



