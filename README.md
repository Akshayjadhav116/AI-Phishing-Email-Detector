# 🛡️ AI-Driven Phishing Email Detection Using NLP

An AI-powered phishing email detection system using **Natural Language Processing (NLP)** and **Machine Learning** to classify emails as **Phishing** or **Legitimate**.

## 🚀 Live Demo

**Streamlit App:**  
https://ai-phishing-email-detector-au3ga6cmvpl5tcyjwyjf5k.streamlit.app/

**GitHub Repository:**  
https://github.com/Akshayjadhav116/AI-Phishing-Email-Detector

## 📌 Project Overview

Phishing emails are fraudulent messages designed to trick users into revealing sensitive information such as passwords, banking details, OTPs, or account credentials.

This project analyzes email text using NLP techniques, converts the text into numerical features using **TF-IDF**, and applies machine learning models to classify emails.

## 🎯 Objectives

- Detect phishing emails automatically.
- Classify emails as Phishing or Legitimate.
- Apply NLP techniques for text preprocessing.
- Convert email text into numerical representations using TF-IDF.
- Compare multiple machine learning models.
- Build an interactive web application using Streamlit.
- Deploy the application on Streamlit Community Cloud.

## 🧠 Technologies Used

- **Python**
- **NLP**
- **TF-IDF**
- **Scikit-learn**
- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Random Forest**
- **Multi-Layer Perceptron (MLP)**
- **Pandas**
- **NumPy**
- **Joblib**
- **Streamlit**
- **Jupyter Notebook**
- **VS Code**
- **Git & GitHub**
- **Streamlit Community Cloud**

## 🔄 System Workflow

```text
             Email Input
                  │
                  ▼
       ┌─────────────────────┐
       │ Text Preprocessing  │
       └─────────────────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │   TF-IDF Vectorizer │
       └─────────────────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Machine Learning    │
       │      Model          │
       └─────────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Prediction        │
        │                   │
        │ 🛑 Phishing       │
        │       OR          │
        │ ✅ Legitimate     │
        └───────────────────┘
                  │
                  ▼
        Confidence / Result
```

## 📊 Dataset

The project uses a labeled email dataset containing:

- **Total Emails:** 1,200
- **Phishing Emails:** 612
- **Legitimate Emails:** 588

## 🔬 NLP Pipeline

### 1. Text Collection

The system receives:
- Email subject
- Sender email
- Email body

### 2. Text Preprocessing

The email content is cleaned and prepared for feature extraction.

### 3. TF-IDF Feature Extraction

**Term Frequency-Inverse Document Frequency (TF-IDF)** converts the cleaned email text into numerical feature vectors and helps represent words that are important for classification.

### 4. Machine Learning Classification

The extracted features are passed to trained machine learning models to classify the email.

## 🤖 Machine Learning Models

The project compares:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest
4. Neural Network (MLP)

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 95.67% | 97.30% | 94.12% | 95.68% |
| Naive Bayes | 95.67% | 97.30% | 94.12% | 95.68% |
| Neural Network (MLP) | 95.67% | 97.30% | 94.12% | 95.68% |
| Random Forest | 95.33% | 96.64% | 94.12% | 95.36% |

The deployed dashboard currently uses **Logistic Regression** as the active model.

## 🖥️ Streamlit Application

The dashboard provides:

- 📊 Model performance dashboard
- 📧 Email Security Analyzer
- 📝 Email subject input
- 👤 Sender email input
- 📄 Email body input
- 🛡️ Phishing detection
- ✅ Legitimate email detection
- 📈 Model comparison
- 📊 Dataset statistics

## 📂 Project Structure

```text
AI-Driven Phishing Email Detection Using NLP/
│
├── assets/
├── data/
│   └── emails_dataset_cleaned.csv
│
├── models/
│   └── Trained machine learning models
│
├── notebooks/
│   └── Phishing_Email_Detection_NLP.ipynb
│
├── app.py
├── requirements.txt
├── emails_dataset_raw.csv
├── model_comparison_results.csv
└── Comparative_Analysis_Report.docx
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Akshayjadhav116/AI-Phishing-Email-Detector.git
```

### 2. Navigate to the Project

```bash
cd AI-Phishing-Email-Detector
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment on Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

```text
Local Development
       │
       ▼
      Git
       │
       ▼
    GitHub
       │
       ▼
Streamlit Community Cloud
       │
       ▼
   Live Web App
```

## 🔐 Security Note

This project is intended for educational and demonstration purposes.

The prediction generated by the model should not be considered a guaranteed security determination. Users should still follow standard email security practices and verify suspicious messages through trusted channels.

## 🔮 Future Enhancements

- URL analysis and malicious link detection
- HTML email analysis
- Sender/domain reputation analysis
- Attachment-based threat detection
- Deep learning models such as LSTM or BERT
- Explainable AI for prediction reasoning
- Real-time email integration
- Larger and more diverse datasets
- Advanced phishing campaign detection
- Improved model calibration and confidence analysis

## 🎓 Academic Project

This project was developed as an academic/technical project focused on:

**Artificial Intelligence + Machine Learning + Natural Language Processing**

## 👨‍💻 Author

**Akshay Jadhav**

B.Tech Computer Science Engineering  
Specialization: Artificial Intelligence & Machine Learning

### Connect

- GitHub: https://github.com/Akshayjadhav116
- LinkedIn: https://www.linkedin.com/in/akshay-jadhav-907431297

## ⭐ Project Highlights

- 🧠 NLP-based phishing detection
- 🔤 TF-IDF feature extraction
- 🤖 Multiple ML models compared
- 📊 1,200 labeled emails
- 🎯 95.67% best accuracy
- 📈 95.68% best F1 score
- 🖥️ Interactive Streamlit dashboard
- ☁️ Cloud deployment
- 🔗 Public GitHub repository

## 📜 License

This project is intended for educational and demonstration purposes.
