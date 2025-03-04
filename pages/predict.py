import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 모델 정의
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(15, 256),  # Updated input size to 15
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# 데이터 전처리 함수
def preprocess_data(data, numeric_features, categorical_features, preprocessor=None):
    # 결측치 처리
    imputer = SimpleImputer(strategy='mean')
    data[numeric_features] = imputer.fit_transform(data[numeric_features])

    if preprocessor is None:
        # 전처리기를 구성: 수치형 데이터에는 StandardScaler, 범주형 데이터에는 OneHotEncoder 적용
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        preprocessor.fit(data)

    # 데이터 전처리
    preprocessed_data = preprocessor.transform(data)
    
    # 전처리된 데이터를 DataFrame으로 변환
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    preprocessed_df.reset_index(drop=True, inplace=True)
    return preprocessed_df, preprocessor

# 더미 고객 데이터 생성
dummy_customers = pd.DataFrame({
    'Country': ['France', 'Spain', 'Germany'],
    'Gender': ['Female', 'Male', 'Female'],
    'Credit Card': [1, 0, 1],
    'Active Member': [1, 0, 1],
    'Credit Score': [600, 700, 800],
    'Age': [40, 50, 60],
    'Tenure': [3, 4, 5],
    'Balance': [60000, 70000, 80000],
    'Products Number': [2, 1, 3],
    'Estimated Salary': [50000, 60000, 70000]
})

# 전처리할 특성 정의
numeric_features = ["Credit Score", "Age", "Tenure", "Balance", "Products Number", "Estimated Salary"]
categorical_features = ["Country", "Gender", "Credit Card", "Active Member"]

# 더미 고객 데이터 전처리
preprocessed_dummy, preprocessor = preprocess_data(dummy_customers, numeric_features, categorical_features)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChurnModel().to(device)
model.load_state_dict(torch.load("churn_model.pth", map_location=device))
model.eval()

# 페이지 설정
st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("🏦 가상 고객 이탈 예측")
st.markdown("---")

# 가상 고객 정보 입력
st.header("🧑‍💼 가상 고객 정보 입력")

customer_info = {}
customer_info["Credit Score"] = st.number_input("📊 신용점수", min_value=0, max_value=1000, value=650)
customer_info["Country"] = st.selectbox("🌍 거주 국가", ["France", "Spain", "Germany"])
customer_info["Gender"] = st.selectbox("⚧ 성별", ["Male", "Female"])
customer_info["Age"] = st.number_input("👤 나이", min_value=18, max_value=100, value=30)
customer_info["Tenure"] = st.number_input("📅 은행 이용 기간(년)", min_value=0, max_value=10, value=5)
customer_info["Balance"] = st.number_input("💰 계좌 잔액", min_value=0.0, value=50000.0)
customer_info["Products Number"] = st.number_input("🛍 보유 상품 수", min_value=1, max_value=10, value=2)
customer_info["Credit Card"] = st.selectbox("💳 신용카드 보유 여부", [0, 1])
customer_info["Active Member"] = st.selectbox("🟢 활성 회원 여부", [0, 1])
customer_info["Estimated Salary"] = st.number_input("💵 예상 연봉", min_value=0.0, value=50000.0)

# 입력된 데이터를 데이터프레임으로 변환
input_df = pd.DataFrame([customer_info])

# 데이터 전처리
preprocessed_input, _ = preprocess_data(input_df, numeric_features, categorical_features, preprocessor)

# 예측 수행
input_tensor = torch.tensor(preprocessed_input.values).float().to(device)
with torch.no_grad():
    prediction = model(input_tensor).item()

# 예측 결과 출력
st.subheader("🔮 예측 결과")
st.write(f"이 고객의 이탈 확률은 {prediction:.2f} 입니다.")
