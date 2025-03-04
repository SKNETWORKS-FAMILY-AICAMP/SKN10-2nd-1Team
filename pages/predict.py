import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Streamlit 페이지 설정
st.set_page_config(page_title="은행 고객 이탈 예측", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .stSlider {
        color: #0073e6;
    }
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 은행 고객 이탈 예측")
st.markdown("---")

# 파일 경로
file_path = "Bank Customer Churn Prediction.csv"

# 데이터 불러오기
df = pd.read_csv(file_path)

# 컬럼명 매핑
df.columns = ["Customer ID", "Credit Score", "Country", "Gender", "Age", "Tenure", "Balance", "Products Number", "Credit Card", "Active Member", "Estimated Salary", "Churn"]

# 모델 로드
model = joblib.load('random_forest_model.pkl')

# 사용자 입력 받기
st.header("🔍 고객 정보 입력")

country = st.selectbox("🌍 거주 국가", df["Country"].unique())
gender = st.selectbox("⚧ 성별", df["Gender"].unique())
credit_score = st.slider("📊 신용점수", int(df["Credit Score"].min()), int(df["Credit Score"].max()), int(df["Credit Score"].mean()))
age = st.slider("👤 나이", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].mean()))
tenure = st.slider("📅 은행 이용 기간(년)", int(df["Tenure"].min()), int(df["Tenure"].max()), int(df["Tenure"].mean()))
balance = st.slider("💰 계좌 잔액", float(df["Balance"].min()), float(df["Balance"].max()), float(df["Balance"].mean()))
products_number = st.slider("🛍 보유 상품 수", int(df["Products Number"].min()), int(df["Products Number"].max()), int(df["Products Number"].mean()))
credit_card = st.radio("💳 신용카드 보유 여부", [0, 1])
active_member = st.radio("🟢 활성 회원 여부", [0, 1])
estimated_salary = st.slider("💵 예상 연봉", float(df["Estimated Salary"].min()), float(df["Estimated Salary"].max()), float(df["Estimated Salary"].mean()))

# 입력 데이터를 DataFrame으로 변환
input_data = pd.DataFrame({
    "Credit Score": [credit_score],
    "Country": [country],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "Products Number": [products_number],
    "Credit Card": [credit_card],
    "Active Member": [active_member],
    "Estimated Salary": [estimated_salary]
})

# 데이터 전처리
categorical_features = ["Country", "Gender"]
numeric_features = ["Credit Score", "Age", "Tenure", "Balance", "Products Number", "Estimated Salary"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
input_data_processed = pipeline.fit_transform(input_data)

# 모델 예측
churn_probability = model.predict_proba(input_data_processed)[:, 1][0]

# 예측 결과 출력
st.subheader("📊 예측 결과")
st.write(f"해당 고객의 이탈 확률은 **{churn_probability:.2f}** 입니다.")