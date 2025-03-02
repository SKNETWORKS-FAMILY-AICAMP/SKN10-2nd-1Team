import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer, StandardScaler

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
model_path = "./model/randomforest_model.pkl"

# 데이터 불러오기
df = pd.read_csv(file_path)
model = pickle.load(open(model_path, 'rb'))

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
    #with col5:
    #    churn = st.radio("🔄 고객 이탈 여부", [0, 1])
    
    credit_score = st.slider("📊 신용점수", int(df["credit_score"].min()), int(df["credit_score"].max()), int(df["credit_score"].mean()))
    age = st.slider("👤 나이", int(df["age"].min()), int(df["age"].max()), int(df["age"].mean()))
    tenure = st.slider("📅 은행 이용 기간(년)", int(df["tenure"].min()), int(df["tenure"].max()), int(df["tenure"].mean()))
    balance = st.slider("💰 계좌 잔액", float(df["balance"].min()), float(df["balance"].max()), float(df["balance"].mean()))
    products_number = st.slider("🛍 보유 상품 수", int(df["products_number"].min()), int(df["products_number"].max()), int(df["products_number"].mean()))
    estimated_salary = st.slider("💵 예상 연봉", float(df["estimated_salary"].min()), float(df["estimated_salary"].max()), float(df["estimated_salary"].mean()))

# 사용자 입력 데이터 필터링
filtered_df = df[
    (df['country'] == country) &
    (df['gender'] == gender) &
    (df['credit_card'] == credit_card) &
    (df['active_member'] == active_member) &
    (df['credit_score'] <= credit_score) &
    (df['age'] <= age) &
    (df['tenure'] <= tenure) &
    (df['balance'] <= balance) &
    (df['products_number'] == products_number) &
    (df['estimated_salary'] <= estimated_salary)
]

# 필터링된 데이터 표시
st.subheader("📊 필터링된 데이터")
st.dataframe(filtered_df)

# 사용자 입력 데이터 전처리
pt1, pt2 = PowerTransformer(method='yeo-johnson'), PowerTransformer(method='yeo-johnson')
pt1.fit_transform(df['credit_score'].values.reshape(-1, 1))
pt2.fit_transform(df['age'].values.reshape(-1, 1))
scaler1, scaler2, scaler3, scaler4 = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
scaler1.fit_transform(df[['credit_score']])
scaler2.fit_transform(df[['age']])
scaler3.fit_transform(df[['balance']])
scaler4.fit_transform(df[['estimated_salary']])

input_data = pd.DataFrame({
    'credit_score': [scaler1.transform(pt1.transform(credit_score))],
    'gender': [1 if gender == 'Male' else 0],
    'age': [scaler2.transform(pt2.transform(age))],
    'tenure': [tenure],
    'balance': [scaler3.transform(balance)],
    'products_number': [products_number],
    'credit_card': [credit_card],
    'active_member': [active_member],
    'estimated_salary': [scaler4.transform(estimated_salary)],
    'country_France': [1 if country == 'France' else 0],
    'country_Germany': [1 if country == 'Germany' else 0],
    'country_Spain': [1 if country == 'Spain' else 0],
})

y_pred = model.predict(input_data)
y_pred_proba = model.predict_proba(input_data)[:, 1]

# 예측 결과 표시
st.subheader("📈 예측 결과")
st.write("고객 이탈 예측: ", "이탈" if y_pred == 1 else "비이탈")
st.write("고객 이탈 예측 확률: ", y_pred_proba)
