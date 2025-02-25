import streamlit as st
import pandas as pd

# 스타일 설정
st.set_page_config(page_title="은행 고객 이탈 예측", layout="wide")
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

# 필터링된 데이터 출력          ,필터링이 제대로 이루어지도록 해야함
st.subheader("📊 필터링된 데이터")
st.dataframe(df.style.set_properties(**{"background-color": "#f9f9f9", "border": "1px solid #ddd", "color": "black"}))

# 중앙 정렬을 위한 컨테이너
with st.container():
    st.header("🔍 필터 옵션")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        country = st.radio("🌍 거주 국가", df["Country"].unique())
    with col2:
        gender = st.radio("⚧ 성별", df["Gender"].unique())
    with col3:
        credit_card = st.radio("💳 신용카드 보유 여부", ["No", "Yes"], index=1)
    with col4:
        active_member = st.radio("🟢 활성 회원 여부", ["No", "Yes"], index=1)
    with col5:
        churn = st.radio("🔄 고객 이탈 여부", ["No", "Yes"], index=0)
    
    credit_score_range = st.slider("📊 신용점수", int(df["Credit Score"].min()), int(df["Credit Score"].max()), (int(df["Credit Score"].min()), int(df["Credit Score"].max())))
    age_range = st.slider("👤 나이", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    tenure_range = st.slider("📅 은행 이용 기간(년)", int(df["Tenure"].min()), int(df["Tenure"].max()), (int(df["Tenure"].min()), int(df["Tenure"].max())))
    balance_range = st.slider("💰 계좌 잔액", float(df["Balance"].min()), float(df["Balance"].max()), (float(df["Balance"].min()), float(df["Balance"].max())))
    products_number_range = st.slider("🛍 보유 상품 수", int(df["Products Number"].min()), int(df["Products Number"].max()), (int(df["Products Number"].min()), int(df["Products Number"].max())))
    estimated_salary_range = st.slider("💵 예상 연봉", float(df["Estimated Salary"].min()), float(df["Estimated Salary"].max()), (float(df["Estimated Salary"].min()), float(df["Estimated Salary"].max())))

filtered_df = df[(df["Credit Score"] >= credit_score_range[0]) & (df["Credit Score"] <= credit_score_range[1]) &
                 (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
                 (df["Tenure"] >= tenure_range[0]) & (df["Tenure"] <= tenure_range[1]) &
                 (df["Balance"] >= balance_range[0]) & (df["Balance"] <= balance_range[1]) &
                 (df["Products Number"] >= products_number_range[0]) & (df["Products Number"] <= products_number_range[1]) &
                 (df["Estimated Salary"] >= estimated_salary_range[0]) & (df["Estimated Salary"] <= estimated_salary_range[1]) &
                 (df["Country"] == country) & (df["Gender"] == gender) &
                 (df["Credit Card"] == (1 if credit_card == "Yes" else 0)) &
                 (df["Active Member"] == (1 if active_member == "Yes" else 0)) &
                 (df["Churn"] == (1 if churn == "Yes" else 0))]

