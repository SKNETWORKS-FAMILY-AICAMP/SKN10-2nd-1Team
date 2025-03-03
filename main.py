import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc

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
file_path = "./data/Bank Customer Churn Prediction.csv"

# 데이터 불러오기
df = pd.read_csv(file_path)

# 중앙 정렬을 위한 컨테이너
with st.container():
    st.header("🔍 필터 옵션")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        country = st.radio("🌍 거주 국가", df["country"].unique())
    with col2:
        gender = st.radio("⚧ 성별", df["gender"].unique())
    with col3:
        credit_card = st.radio("💳 신용카드 보유 여부", [0, 1])
    with col4:
        active_member = st.radio("🟢 활성 회원 여부", [0, 1])
    with col5:
        churn = st.radio("🔄 고객 이탈 여부", [0, 1])
    
    credit_score_range = st.slider("📊 신용점수", int(df["credit_score"].min()), int(df["credit_score"].max()), (int(df["credit_score"].min()), int(df["credit_score"].max())))
    age_range = st.slider("👤 나이", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
    tenure_range = st.slider("📅 은행 이용 기간(년)", int(df["tenure"].min()), int(df["tenure"].max()), (int(df["tenure"].min()), int(df["tenure"].max())))
    balance_range = st.slider("💰 계좌 잔액", float(df["balance"].min()), float(df["balance"].max()), (float(df["balance"].min()), float(df["balance"].max())))
    products_number_range = st.slider("🛍 보유 상품 수", int(df["products_number"].min()), int(df["products_number"].max()), (int(df["products_number"].min()), int(df["products_number"].max())))
    estimated_salary_range = st.slider("💵 예상 연봉", float(df["estimated_salary"].min()), float(df["estimated_salary"].max()), (float(df["estimated_salary"].min()), float(df["estimated_salary"].max())))

filtered_df = df[(df["credit_score"] >= credit_score_range[0]) & (df["credit_score"] <= credit_score_range[1]) &
                (df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) &
                (df["tenure"] >= tenure_range[0]) & (df["tenure"] <= tenure_range[1]) &
                (df["balance"] >= balance_range[0]) & (df["balance"] <= balance_range[1]) &
                (df["products_number"] >= products_number_range[0]) & (df["products_number"] <= products_number_range[1]) &
                (df["estimated_salary"] >= estimated_salary_range[0]) & (df["estimated_salary"] <= estimated_salary_range[1]) &
                (df["country"] == country) & (df["gender"] == gender) &
                (df["credit_card"] == credit_card) &
                (df["active_member"] == active_member) &
                (df["churn"] == churn)]

# 필터링된 데이터 출력
st.subheader("📊 필터링된 데이터")
st.dataframe(filtered_df.style.set_properties(**{"background-color": "#f9f9f9", "border": "1px solid #ddd", "color": "black"}))

# 전처리
filtered_df['country_France'] = filtered_df['country'].apply(lambda x: 1 if x == 'France' else 0)
filtered_df['country_Germany'] = filtered_df['country'].apply(lambda x: 1 if x == 'Germany' else 0)
filtered_df['country_Spain'] = filtered_df['country'].apply(lambda x: 1 if x == 'Spain' else 0)
filtered_df.drop('country', axis=1, inplace=True)

filtered_df['gender'] = filtered_df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

pt = PowerTransformer(method='yeo-johnson')
pt.fit_transform(df['credit_score'].values.reshape(-1, 1))
filtered_df['credit_score'] = pt.transform(filtered_df['credit_score'].values.reshape(-1,1))
pt.fit_transform(df['age'].values.reshape(-1, 1))
filtered_df['age'] = pt.transform(filtered_df['age'].values.reshape(-1,1))

scaler = StandardScaler()
scaler.fit_transform(df['credit_score'].values.reshape(-1, 1))
filtered_df['credit_score'] = scaler.transform(filtered_df['credit_score'].values.reshape(-1,1))
scaler.fit_transform(df['age'].values.reshape(-1, 1))
filtered_df['age'] = scaler.transform(filtered_df['age'].values.reshape(-1,1))
scaler.fit_transform(df['balance'].values.reshape(-1, 1))
filtered_df['balance'] = scaler.transform(filtered_df['balance'].values.reshape(-1,1))
scaler.fit_transform(df['estimated_salary'].values.reshape(-1, 1))
filtered_df['estimated_salary'] = scaler.transform(filtered_df['estimated_salary'].values.reshape(-1,1))

# 모델 불러오기
model = pickle.load(open("./model/randomforest_model.pkl", "rb"))

# 예측
y = filtered_df["churn"]
X = filtered_df.drop(["churn", "customer_id"], axis=1)

y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# 결과 출력
st.subheader("📈 예측 결과")
st.write(f"예측 결과: {y_pred}")
st.write(f'실제값: {y.values}')
st.write(f"예측 확률: {y_pred_proba}")
