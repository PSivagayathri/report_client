import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from prophet import Prophet
import spacy
import time
import requests
import json
import joblib  # ✅ to load your trained model and vectorizer

# ------------------------------
# Backend URLs
# ------------------------------
AUTH_URL = "http://127.0.0.1:8000/api/auth"
FINANCE_URL = "http://127.0.0.1:8000/api/finance"

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="AI Financial Report Analysis", layout="wide")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# ------------------------------
# Authentication UI
# ------------------------------
def login_ui():
    st.title("🔐 Login to AI Financial Report Analysis")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email and password:
            try:
                response = requests.post(f"{AUTH_URL}/login", json={"email": email, "password": password})
                if response.status_code == 200:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success("✅ Logged in successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Invalid credentials"))
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to backend. Please start your FastAPI server.")
        else:
            st.error("❌ Please enter valid email and password.")

    st.write("Don't have an account?")
    if st.button("Go to Signup"):
        st.session_state.show_signup = True
        st.rerun()


def signup_ui():
    st.title("📝 Create an Account")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if not name or not email or not password:
            st.warning("⚠️ Please fill in all fields.")
        elif password != confirm:
            st.error("❌ Passwords do not match.")
        else:
            try:
                response = requests.post(f"{AUTH_URL}/signup",
                                         json={"name": name, "email": email, "password": password})
                if response.status_code == 200:
                    st.success("✅ Account created successfully! Please log in.")
                    time.sleep(1)
                    st.session_state.show_signup = False
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Signup failed"))
            except:
                st.error("⚠️ Could not connect to backend.")

    st.markdown("---")
    if st.button("Back to Login"):
        st.session_state.show_signup = False
        st.rerun()

# ------------------------------
# Auth Routing
# ------------------------------
if not st.session_state.authenticated:
    if st.session_state.show_signup:
        signup_ui()
    else:
        login_ui()
    st.stop()

# ------------------------------
# Logout Button
# ------------------------------
st.sidebar.success(f"👋 Logged in as: {st.session_state.user_email}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.rerun()

# ------------------------------
# Load NLP & Custom Sentiment Model
# ------------------------------
nlp = spacy.load("en_core_web_sm")

# ✅ Load trained model and vectorizer
try:
    model = joblib.load("financial_sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    st.sidebar.info("✅ Custom Sentiment Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading model: {e}")
    model = None
    vectorizer = None

# ------------------------------
# Main App Dashboard
# ------------------------------
st.title("📊 AI-Powered Financial Report Analysis")

st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose Task",
    ["Upload Report", "Market Data & Forecast", "Sentiment Analysis"]
)

# ------------------------------
# Upload & Summarize Reports
# ------------------------------
if option == "Upload Report":
    st.header("📄 Upload Financial Report (CSV, Excel, TXT)")
    uploaded_file = st.file_uploader("Upload Report", type=["csv", "xlsx", "txt"])

    if uploaded_file:
        summary_md = ""
        df = None

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
            doc = nlp(text)

            st.subheader("🔍 Extracted Entities")
            if len(doc.ents) > 0:
                ent_data = [{"Entity": ent.text, "Label": ent.label_} for ent in doc.ents]
                st.dataframe(pd.DataFrame(ent_data))
            else:
                st.info("No named entities found in the text.")

            # Simple keyword-based summary (you can use a transformer summarizer too)
            st.subheader("🧠 Summary (Keyword-based)")
            sentences = text.split(".")
            key_sentences = [s for s in sentences if "revenue" in s or "profit" in s or "growth" in s]
            if key_sentences:
                for s in key_sentences[:5]:
                    st.markdown(f"- {s.strip()}")
            summary_md = " ".join(key_sentences)

        if df is not None:
            st.subheader("📘 Preview Data")
            st.dataframe(df.head())

            if st.button("🪄 Generate Summary from Table"):
                st.subheader("🧾 AI-Generated Summary from Financial Data")
                companies = df["company"].unique() if "company" in df.columns else []
                summary_md = "### 🏢 Company Overview\n"
                if len(companies) > 0:
                    summary_md += f"- **Companies:** {', '.join(companies[:5])}\n"

                st.markdown(summary_md)

        # ✅ Save report summary to backend
        if summary_md.strip() != "":
            save_payload = {
                "email": st.session_state.user_email,
                "report_name": uploaded_file.name,
                "summary": summary_md,
            }
            try:
                response = requests.post(f"{FINANCE_URL}/save_report", json=save_payload)
                if response.status_code == 200:
                    st.success("✅ Report summary saved to backend.")
            except:
                st.warning("⚠️ Could not connect to backend.")

# ------------------------------
# Market Data & Forecast
# ------------------------------
elif option == "Market Data & Forecast":
    st.header("📈 Stock Price Forecasting")

    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")
    forecast_days = st.number_input("Forecast Period (Days)", min_value=7, max_value=365, value=90, step=1)

    if st.button("Fetch & Forecast"):
        try:
            data = yf.download(ticker, period="2y")

            # 🔹 Flatten multi-level columns if present (important fix)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            if data.empty:
                st.warning("⚠️ No data found for this ticker symbol.")
                st.stop()

            close_col = "Close" if "Close" in data.columns else data.columns[0]
            st.dataframe(data.tail())

            # Plot closing prices
            fig = px.line(data, x=data.index, y=close_col, title=f"{ticker} Closing Prices Over Time")
            st.plotly_chart(fig, use_container_width=True)

            # Prophet forecasting
            df_prophet = data.reset_index()[["Date", close_col]].rename(columns={"Date": "ds", close_col: "y"})
            model_prophet = Prophet(daily_seasonality=True)
            model_prophet.fit(df_prophet)

            future = model_prophet.make_future_dataframe(periods=int(forecast_days))
            forecast = model_prophet.predict(future)

            # Plot forecast
            fig2 = px.line(forecast, x="ds", y="yhat", title=f"{ticker} Forecast for {forecast_days} Days")
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error fetching or forecasting data: {e}")


# ------------------------------
# Sentiment Analysis
# ------------------------------
elif option == "Sentiment Analysis":
    st.header("📰 Financial Sentiment Analysis (Custom Model)")
    user_input = st.text_area("Enter financial statement or news excerpt:")

    if st.button("Analyze Sentiment"):
        if model and vectorizer:
            if user_input.strip():
                X_input = vectorizer.transform([user_input])
                pred = model.predict(X_input)[0]

                st.subheader("📈 Sentiment Result")
                if pred == "positive":
                    st.success("✅ Positive Sentiment Detected")
                elif pred == "negative":
                    st.error("❌ Negative Sentiment Detected")
                else:
                    st.info("⚖️ Neutral Sentiment Detected")

                # ✅ Save result to backend
                payload = {
                    "email": st.session_state.user_email,
                    "text": user_input,
                    "sentiment": pred
                }
                try:
                    response = requests.post(f"{FINANCE_URL}/save_sentiment", json=payload)
                    if response.status_code == 200:
                        st.success("✅ Sentiment result saved successfully!")
                except:
                    st.warning("⚠️ Could not connect to backend.")
            else:
                st.warning("Please enter some text.")
        else:
            st.error("⚠️ Custom sentiment model not loaded properly.")
