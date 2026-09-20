# ============================================================
# AI-DRIVEN PHISHING EMAIL DETECTOR
# NLP + Machine Learning + Streamlit Dashboard
# ============================================================

import os
import re
import time
import html
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main Background ---------- */

    .stApp {
        background: #0b0f16;
        color: #f5f7fa;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #111722;
        border-right: 1px solid #263142;
    }

    /* ---------- Header ---------- */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #4da3ff;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        color: #aab4c3;
        font-size: 16px;
        margin-bottom: 35px;
    }

    /* ---------- Cards ---------- */

    .dashboard-card {
        background: #151b25;
        border: 1px solid #293445;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        min-height: 135px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }

    .dashboard-label {
        color: #b7c0ce;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .dashboard-value {
        color: #4da3ff;
        font-size: 30px;
        font-weight: 800;
    }

    .dashboard-small {
        color: #9da8b8;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ---------- Section Titles ---------- */

    .section-title {
        color: #f4f7fb;
        font-size: 24px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* ---------- Result Cards ---------- */

    .phishing-card {
        background: #421d24;
        border-left: 7px solid #ff4d5e;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
    }

    .safe-card {
        background: #123523;
        border-left: 7px solid #20d879;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
    }

    .result-description {
        color: #d7dce5;
        font-size: 15px;
        margin-top: 8px;
    }

    /* ---------- Info Cards ---------- */

    .info-card {
        background: #151b25;
        border: 1px solid #293445;
        border-radius: 14px;
        padding: 20px;
        margin-top: 10px;
    }

    .info-title {
        color: #4da3ff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* ---------- Risk ---------- */

    .risk-high {
        background: #461c24;
        border: 1px solid #ff4d5e;
        color: #ff7180;
        padding: 12px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    .risk-medium {
        background: #463a1b;
        border: 1px solid #eabf42;
        color: #f0ce61;
        padding: 12px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    .risk-low {
        background: #153824;
        border: 1px solid #20d879;
        color: #45e995;
        padding: 12px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 11px;
        border: none;
        background: linear-gradient(
            90deg,
            #2563eb,
            #3b82f6
        );
        color: white;
        font-size: 16px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #1d4ed8;
    }

    /* ---------- Text Area ---------- */

    textarea {
        border-radius: 12px !important;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #788496;
        margin-top: 60px;
        padding: 25px;
        border-top: 1px solid #263142;
    }

    /* ---------- Divider ---------- */

    hr {
        border: 1px solid #263142;
        margin-top: 30px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "emails_dataset_cleaned.csv"
)

ROOT_DATA_PATH = os.path.join(
    BASE_DIR,
    "emails_dataset_cleaned.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# NLTK
# ============================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()

STOP_WORDS = set(
    stopwords.words("english")
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert HTML entities
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Replace URLs with common token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " urltoken ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Keep alphabetic characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Split
    words = text.split()

    # Remove stopwords and stem
    cleaned_words = []

    for word in words:

        if word not in STOP_WORDS:

            cleaned_words.append(
                stemmer.stem(word)
            )

    return " ".join(cleaned_words)


# ============================================================
# FIND DATASET
# ============================================================

def get_dataset_path():

    if os.path.exists(DATA_PATH):
        return DATA_PATH

    if os.path.exists(ROOT_DATA_PATH):
        return ROOT_DATA_PATH

    return None


# ============================================================
# DETECT DATASET COLUMNS
# ============================================================

def find_column(df, possible_names):

    columns_lower = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for name in possible_names:

        if name.lower() in columns_lower:

            return columns_lower[name.lower()]

    return None


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    path = get_dataset_path()

    if path is None:

        return None

    df = pd.read_csv(path)

    # Remove completely empty rows
    df = df.dropna(how="all")

    return df


# ============================================================
# IDENTIFY DATASET STRUCTURE
# ============================================================

def prepare_dataset(df):

    df = df.copy()

    # --------------------------------------------------------
    # Find label
    # --------------------------------------------------------

    label_col = find_column(
        df,
        [
            "label",
            "class",
            "target",
            "category",
            "type",
            "is_phishing",
            "phishing"
        ]
    )

    if label_col is None:

        # Try common last-column approach
        label_col = df.columns[-1]

    # --------------------------------------------------------
    # Find subject
    # --------------------------------------------------------

    subject_col = find_column(
        df,
        [
            "subject",
            "email_subject",
            "mail_subject",
            "title"
        ]
    )

    # --------------------------------------------------------
    # Find body
    # --------------------------------------------------------

    body_col = find_column(
        df,
        [
            "body",
            "email_body",
            "email",
            "message",
            "text",
            "content",
            "mail_body"
        ]
    )

    # --------------------------------------------------------
    # If body not found, select text column
    # --------------------------------------------------------

    if body_col is None:

        text_candidates = []

        for col in df.columns:

            if col == label_col:
                continue

            if df[col].dtype == "object":

                avg_length = (
                    df[col]
                    .astype(str)
                    .str.len()
                    .mean()
                )

                text_candidates.append(
                    (col, avg_length)
                )

        if text_candidates:

            text_candidates.sort(
                key=lambda x: x[1],
                reverse=True
            )

            body_col = text_candidates[0][0]

    # --------------------------------------------------------
    # Create combined text
    # --------------------------------------------------------

    if subject_col is not None:

        df["combined_text"] = (
            df[subject_col].fillna("").astype(str)
            + " "
            + df[body_col].fillna("").astype(str)
        )

    else:

        df["combined_text"] = (
            df[body_col]
            .fillna("")
            .astype(str)
        )

    # --------------------------------------------------------
    # Convert labels
    # --------------------------------------------------------

    original_labels = df[label_col]

    def convert_label(value):

        value_string = str(value).strip().lower()

        # Phishing labels
        if value_string in [
            "phishing",
            "phish",
            "spam",
            "malicious",
            "1",
            "true",
            "yes"
        ]:
            return 1

        # Legitimate labels
        if value_string in [
            "legitimate",
            "legit",
            "ham",
            "safe",
            "normal",
            "0",
            "false",
            "no"
        ]:
            return 0

        # Numeric fallback
        try:

            number = float(value)

            if number == 1:
                return 1

            if number == 0:
                return 0

        except:
            pass

        return np.nan

    df["target"] = original_labels.apply(
        convert_label
    )

    df = df.dropna(
        subset=["combined_text", "target"]
    )

    df["target"] = df["target"].astype(int)

    return df, label_col, subject_col, body_col


# ============================================================
# METADATA FEATURES
# ============================================================

def extract_metadata(text):

    text = str(text)

    # URL count
    url_count = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE
        )
    )

    # Exclamation count
    exclamation_count = text.count("!")

    # HTML presence
    html_presence = int(
        bool(
            re.search(
                r"<[^>]+>",
                text
            )
        )
    )

    # Message length
    message_length = len(text)

    # Free webmail indicator
    free_webmail_domains = [
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "icloud.com",
        "protonmail.com"
    ]

    lower_text = text.lower()

    free_webmail = int(
        any(
            domain in lower_text
            for domain in free_webmail_domains
        )
    )

    return [
        url_count,
        exclamation_count,
        html_presence,
        message_length,
        free_webmail
    ]


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_models():

    df = load_dataset()

    if df is None:

        return None

    prepared = prepare_dataset(df)

    df, label_col, subject_col, body_col = prepared

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    df["cleaned_text"] = (
        df["combined_text"]
        .apply(clean_text)
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = np.array(
        [
            extract_metadata(text)
            for text in df["combined_text"]
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        min_df=2
    )

    X_text = vectorizer.fit_transform(
        df["cleaned_text"]
    )

    # --------------------------------------------------------
    # Scale metadata
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    X_metadata = scaler.fit_transform(
        metadata
    )

    X_metadata = csr_matrix(
        X_metadata
    )

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    X = hstack(
        [
            X_text,
            X_metadata
        ]
    ).tocsr()

    y = df["target"].values

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                C=2.0,
                max_iter=1000
            ),

        "Naive Bayes":
            MultinomialNB(
                alpha=0.5
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),

        "Neural Network (MLP)":
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                random_state=42,
                max_iter=300
            )
    }

    results = []

    trained_models = {}

    # ========================================================
    # TRAIN
    # ========================================================

    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
        )

        trained_models[
            model_name
        ] = model

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # Best model
    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[
        best_model_name
    ]

    # ========================================================
    # SAVE ARTIFACTS
    # ========================================================

    joblib.dump(
        vectorizer,
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.pkl"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            MODEL_DIR,
            "metadata_scaler.pkl"
        )
    )

    joblib.dump(
        best_model,
        os.path.join(
            MODEL_DIR,
            "best_model.pkl"
        )
    )

    results_df.to_csv(
        os.path.join(
            MODEL_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    # Save every model
    for name, model in trained_models.items():

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                safe_name + ".pkl"
            )
        )

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    metadata_names = [
        "URL Count",
        "Exclamation Count",
        "HTML Presence",
        "Message Length",
        "Free Webmail"
    ]

    return {
        "df": df,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "models": trained_models,
        "results": results_df,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "metadata_names": metadata_names,
        "label_col": label_col,
        "subject_col": subject_col,
        "body_col": body_col,
        "X_test": X_test,
        "y_test": y_test
    }


# ============================================================
# LOAD / TRAIN
# ============================================================

with st.spinner(
    "🔄 Loading AI phishing detection system..."
):

    try:

        pipeline = train_models()

    except Exception as error:

        pipeline = None

        st.error(
            "❌ Unable to initialize the ML pipeline."
        )

        st.exception(error)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🛡️ AI Phishing Detector"
    )

    st.caption(
        "NLP-powered email security system"
    )

    st.markdown("---")

    st.success(
        "🟢 System Online"
    )

    if pipeline is not None:

        results_df = pipeline["results"]

        best_model_name = pipeline[
            "best_model_name"
        ]

        best_accuracy = (
            results_df.iloc[0]["Accuracy"] * 100
        )

        dataset_size = len(
            pipeline["df"]
        )

        phishing_count = int(
            pipeline["df"]["target"].sum()
        )

        legitimate_count = (
            dataset_size - phishing_count
        )

        st.metric(
            "🎯 Best Accuracy",
            f"{best_accuracy:.2f}%"
        )

        st.metric(
            "📚 Dataset",
            f"{dataset_size:,}"
        )

        st.metric(
            "🤖 Best Model",
            best_model_name
        )

        st.markdown("---")

        st.markdown(
            "### 📊 Dataset"
        )

        st.write(
            f"🔴 Phishing: **{phishing_count:,}**"
        )

        st.write(
            f"🟢 Legitimate: **{legitimate_count:,}**"
        )

        st.markdown("---")

    st.markdown(
        "### 🧠 Technology Stack"
    )

    st.write(
        "✅ Python"
    )

    st.write(
        "✅ NLP"
    )

    st.write(
        "✅ TF-IDF"
    )

    st.write(
        "✅ Scikit-learn"
    )

    st.write(
        "✅ Logistic Regression"
    )

    st.write(
        "✅ Random Forest"
    )

    st.write(
        "✅ Streamlit"
    )

    st.markdown("---")

    st.info(
        """
        This system analyzes email
        language and structural features
        to identify potentially phishing
        messages.
        """
    )

    st.markdown("---")

    st.caption(
        "Developed by Akshay Jadhav"
    )

    st.caption(
        "AI & Machine Learning"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ AI-Driven Phishing Email Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Detect phishing emails using Natural Language Processing, '
    'TF-IDF and Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK PIPELINE
# ============================================================

if pipeline is None:

    st.error(
        """
        Dataset or ML pipeline could not be loaded.

        Make sure this file exists:

        `data/emails_dataset_cleaned.csv`
        """
    )

    st.stop()


# ============================================================
# DASHBOARD METRICS
# ============================================================

results_df = pipeline["results"]

dataset_size = len(
    pipeline["df"]
)

best_model_name = pipeline[
    "best_model_name"
]

best_accuracy = (
    results_df.iloc[0]["Accuracy"] * 100
)

best_f1 = (
    results_df.iloc[0]["F1 Score"] * 100
)

phishing_count = int(
    pipeline["df"]["target"].sum()
)

phishing_percentage = (
    phishing_count / dataset_size
) * 100


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="dashboard-card">

        <div class="dashboard-label">
        🎯 Best Accuracy
        </div>

        <div class="dashboard-value">
        {best_accuracy:.2f}%
        </div>

        <div class="dashboard-small">
        Model Performance
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="dashboard-card">

        <div class="dashboard-label">
        📚 Dataset
        </div>

        <div class="dashboard-value">
        {dataset_size:,}
        </div>

        <div class="dashboard-small">
        Labeled Emails
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="dashboard-card">

        <div class="dashboard-label">
        🤖 Best Model
        </div>

        <div class="dashboard-value"
        style="font-size:22px;">
        {best_model_name}
        </div>

        <div class="dashboard-small">
        Active ML Model
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="dashboard-card">

        <div class="dashboard-label">
        📈 Best F1 Score
        </div>

        <div class="dashboard-value">
        {best_f1:.2f}%
        </div>

        <div class="dashboard-small">
        Balanced Performance
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")


# ============================================================
# EMAIL ANALYZER
# ============================================================

st.markdown(
    '<div class="section-title">📧 Email Security Analyzer</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter an email subject and body below. "
    "The AI model will analyze the message."
)


col_subject, col_sender = st.columns(2)


with col_subject:

    subject = st.text_input(
        "📌 Email Subject",
        placeholder="Example: Urgent: Verify your account immediately"
    )


with col_sender:

    sender = st.text_input(
        "👤 Sender Email",
        placeholder="Example: support@example.com"
    )


email_body = st.text_area(
    "📝 Email Body",
    height=260,
    placeholder=(
        "Paste the complete email message here..."
    )
)


# ============================================================
# SAMPLE EMAILS
# ============================================================

sample_col1, sample_col2, sample_col3 = st.columns(3)


with sample_col1:

    if st.button(
        "📨 Load Phishing Example"
    ):

        st.session_state[
            "sample_subject"
        ] = "Urgent: Verify Your Account Now"

        st.session_state[
            "sample_body"
        ] = """
Dear Customer,

Your account has been temporarily suspended.

You must verify your account immediately
to prevent permanent deactivation.

Click the link below to verify your information:

http://secure-account-verification.example.com

Failure to complete verification within
24 hours will result in account closure.

Thank you.
Security Team
"""


with sample_col2:

    if st.button(
        "📩 Load Legitimate Example"
    ):

        st.session_state[
            "sample_subject"
        ] = "Team Meeting Scheduled for Tomorrow"

        st.session_state[
            "sample_body"
        ] = """
Hello Team,

This is a reminder that our weekly team meeting
is scheduled for tomorrow at 10:00 AM.

Please review the agenda before the meeting.

Regards,
Team Coordinator
"""


with sample_col3:

    if st.button(
        "🗑️ Clear Email"
    ):

        st.session_state[
            "sample_subject"
        ] = ""

        st.session_state[
            "sample_body"
        ] = ""


# ============================================================
# USE SESSION SAMPLE VALUES
# ============================================================

if "sample_subject" in st.session_state:

    subject = st.session_state[
        "sample_subject"
    ]

if "sample_body" in st.session_state:

    email_body = st.session_state[
        "sample_body"
    ]


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

analyze = st.button(
    "🔍 ANALYZE EMAIL",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not email_body.strip():

        st.warning(
            "⚠️ Please enter an email body."
        )

    else:

        start_time = time.time()

        # ----------------------------------------------------
        # Combine subject + body
        # ----------------------------------------------------

        complete_email = (
            str(subject)
            + " "
            + str(email_body)
        )

        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        cleaned = clean_text(
            complete_email
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        vectorizer = pipeline[
            "vectorizer"
        ]

        scaler = pipeline[
            "scaler"
        ]

        model = pipeline[
            "best_model"
        ]

        text_vector = vectorizer.transform(
            [cleaned]
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = np.array(
            [
                extract_metadata(
                    complete_email
                )
            ],
            dtype=float
        )

        metadata_scaled = scaler.transform(
            metadata
        )

        metadata_sparse = csr_matrix(
            metadata_scaled
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        final_vector = hstack(
            [
                text_vector,
                metadata_sparse
            ]
        ).tocsr()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            final_vector
        )[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                final_vector
            )[0]

            phishing_probability = (
                probabilities[1] * 100
            )

            legitimate_probability = (
                probabilities[0] * 100
            )

        else:

            decision = model.decision_function(
                final_vector
            )[0]

            phishing_probability = (
                1 /
                (
                    1 +
                    np.exp(-decision)
                )
            ) * 100

            legitimate_probability = (
                100 -
                phishing_probability
            )

        confidence = max(
            phishing_probability,
            legitimate_probability
        )

        prediction_time = (
            time.time() -
            start_time
        )

        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🚨 Detection Result'
            '</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.markdown(
                f"""
                <div class="phishing-card">

                <div class="result-title">
                🚨 PHISHING EMAIL
                </div>

                <div class="result-description">
                The AI model has identified this
                email as potentially malicious or
                phishing-related.
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="safe-card">

                <div class="result-title">
                ✅ LEGITIMATE EMAIL
                </div>

                <div class="result-description">
                The AI model considers this email
                likely to be legitimate.
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # RESULT METRICS
        # ====================================================

        st.write("")

        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "🎯 Confidence",
                f"{confidence:.2f}%"
            )


        with r2:

            st.metric(
                "⚠️ Phishing Probability",
                f"{phishing_probability:.2f}%"
            )


        with r3:

            st.metric(
                "⚡ Prediction Time",
                f"{prediction_time:.4f} sec"
            )


        # ====================================================
        # RISK LEVEL
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '⚠️ Risk Assessment'
            '</div>',
            unsafe_allow_html=True
        )


        if phishing_probability >= 75:

            st.markdown(
                """
                <div class="risk-high">
                🔴 HIGH RISK — Strong phishing indicators detected
                </div>
                """,
                unsafe_allow_html=True
            )

        elif phishing_probability >= 40:

            st.markdown(
                """
                <div class="risk-medium">
                🟠 MEDIUM RISK — Review this email carefully
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="risk-low">
                🟢 LOW RISK — No strong phishing signal detected
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILITY BARS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Probability'
            '</div>',
            unsafe_allow_html=True
        )

        p1, p2 = st.columns(2)


        with p1:

            st.markdown(
                "### 🟢 Legitimate"
            )

            st.progress(
                int(
                    legitimate_probability
                )
            )

            st.write(
                f"{legitimate_probability:.2f}%"
            )


        with p2:

            st.markdown(
                "### 🔴 Phishing"
            )

            st.progress(
                int(
                    phishing_probability
                )
            )

            st.write(
                f"{phishing_probability:.2f}%"
            )


        # ====================================================
        # EMAIL INDICATORS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔎 Email Indicators'
            '</div>',
            unsafe_allow_html=True
        )


        indicators = extract_metadata(
            complete_email
        )


        i1, i2, i3, i4, i5 = st.columns(5)


        with i1:

            st.metric(
                "🔗 URLs",
                indicators[0]
            )


        with i2:

            st.metric(
                "❗ Exclamations",
                indicators[1]
            )


        with i3:

            st.metric(
                "🌐 HTML",
                "Yes" if indicators[2] else "No"
            )


        with i4:

            st.metric(
                "📏 Message Length",
                indicators[3]
            )


        with i5:

            st.metric(
                "📧 Free Webmail",
                "Yes" if indicators[4] else "No"
            )


        # ====================================================
        # SUSPICIOUS KEYWORDS
        # ====================================================

        suspicious_keywords = [
            "urgent",
            "verify",
            "verification",
            "password",
            "account",
            "suspended",
            "click",
            "login",
            "security",
            "confirm",
            "winner",
            "prize",
            "payment",
            "bank",
            "credit",
            "limited",
            "expire",
            "immediately",
            "claim",
            "refund"
        ]


        lower_email = complete_email.lower()

        detected_keywords = []

        for keyword in suspicious_keywords:

            if keyword in lower_email:

                detected_keywords.append(
                    keyword
                )


        # ====================================================
        # AI INSIGHTS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🧠 AI Security Insights'
            '</div>',
            unsafe_allow_html=True
        )


        insight_col1, insight_col2 = st.columns(2)


        with insight_col1:

            st.markdown(
                """
                <div class="info-card">

                <div class="info-title">
                🔍 Suspicious Keywords
                </div>

                """,
                unsafe_allow_html=True
            )

            if detected_keywords:

                st.write(
                    ", ".join(
                        detected_keywords
                    )
                )

            else:

                st.write(
                    "No major suspicious keywords detected."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        with insight_col2:

            st.markdown(
                """
                <div class="info-card">

                <div class="info-title">
                🛡️ Security Recommendation
                </div>

                """,
                unsafe_allow_html=True
            )

            if prediction == 1:

                st.write(
                    "Do not click links, "
                    "download attachments, or "
                    "provide sensitive information."
                )

            else:

                st.write(
                    "The message appears legitimate, "
                    "but always verify unexpected "
                    "requests independently."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '📈 Machine Learning Model Comparison'
    '</div>',
    unsafe_allow_html=True
)

display_results = results_df.copy()

display_results[
    "Accuracy"
] = (
    display_results["Accuracy"] * 100
).round(2)

display_results[
    "Precision"
] = (
    display_results["Precision"] * 100
).round(2)

display_results[
    "Recall"
] = (
    display_results["Recall"] * 100
).round(2)

display_results[
    "F1 Score"
] = (
    display_results["F1 Score"] * 100
).round(2)


display_results = display_results.rename(
    columns={
        "Accuracy": "Accuracy (%)",
        "Precision": "Precision (%)",
        "Recall": "Recall (%)",
        "F1 Score": "F1 Score (%)"
    }
)


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL PERFORMANCE BAR CHART
# ============================================================

st.markdown(
    "### 📊 Performance Visualization"
)

chart_data = display_results.set_index(
    "Model"
)[
    [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)"
    ]
]

st.bar_chart(
    chart_data
)


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '📚 Dataset Overview'
    '</div>',
    unsafe_allow_html=True
)


d1, d2 = st.columns(2)


with d1:

    st.metric(
        "📨 Total Emails",
        f"{dataset_size:,}"
    )


with d2:

    st.metric(
        "🔴 Phishing Percentage",
        f"{phishing_percentage:.2f}%"
    )


distribution_df = pd.DataFrame(
    {
        "Class": [
            "Legitimate",
            "Phishing"
        ],
        "Count": [
            dataset_size - phishing_count,
            phishing_count
        ]
    }
)

distribution_df = distribution_df.set_index(
    "Class"
)


st.bar_chart(
    distribution_df
)


# ============================================================
# NLP PIPELINE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '🧠 NLP Detection Pipeline'
    '</div>',
    unsafe_allow_html=True
)


step1, step2, step3, step4 = st.columns(4)


with step1:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-title">
        1️⃣ Preprocessing
        </div>

        HTML removal<br>
        Stopword removal<br>
        Stemming<br>
        URL normalization

        </div>
        """,
        unsafe_allow_html=True
    )


with step2:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-title">
        2️⃣ TF-IDF
        </div>

        Unigrams<br>
        Bigrams<br>
        3,000 features<br>
        Text representation

        </div>
        """,
        unsafe_allow_html=True
    )


with step3:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-title">
        3️⃣ Metadata
        </div>

        URL count<br>
        Message length<br>
        HTML presence<br>
        Exclamation count

        </div>
        """,
        unsafe_allow_html=True
    )


with step4:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-title">
        4️⃣ Classification
        </div>

        Logistic Regression<br>
        Naive Bayes<br>
        Random Forest<br>
        Neural Network

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    'ℹ️ About This Project'
    '</div>',
    unsafe_allow_html=True
)


about1, about2 = st.columns(2)


with about1:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-title">
        🛡️ Project Objective
        </div>

        This system uses Natural Language Processing
        and Machine Learning to automatically classify
        emails as phishing or legitimate.

        The model analyzes linguistic patterns and
        structural characteristics of email messages.

        </div>
        """,
        unsafe_allow_html=True
    )


with about2:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-title">
        🔬 Feature Engineering
        </div>

        The project combines TF-IDF text features
        with structural metadata such as URL count,
        message length, HTML presence, exclamation
        count and free-webmail indicators.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🛡️ <b>AI-Driven Phishing Email Detection Using NLP</b>

    <br><br>

    Machine Learning • Natural Language Processing • Cybersecurity

    <br><br>

    Developed by <b>Akshay Jadhav</b>

    </div>
    """,
    unsafe_allow_html=True
)