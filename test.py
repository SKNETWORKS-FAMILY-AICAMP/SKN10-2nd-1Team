import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 범주형과 수치형 특성 정의
categorical_features = ['country', 'gender', 'credit_card', 'active_member']
numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

# 모델 정의
class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
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

def predict_customer_churn(model, customer_data, numeric_features, categorical_features, preprocessor):
    # 데이터 전처리
    preprocessed_data = preprocessor.transform(customer_data)
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    print(preprocessed_df.shape)  # 전처리된 데이터의 열 수 확인
    
    # 텐서로 변환
    customer_tensor = torch.tensor(preprocessed_df.values).float().to(device)
    
    # 예측
    with torch.no_grad():
        output = model(customer_tensor)
        churn_probability = output.squeeze().item()
    
    return churn_probability

# 더미 고객 데이터
dummy_customers = pd.DataFrame({
    'country': ['France', 'Spain', 'Germany'],
    'gender': ['Female', 'Male', 'Female'],
    'credit_card': [1, 0, 1],
    'active_member': [1, 0, 1],
    'credit_score': [600, 700, 800],
    'age': [40, 50, 60],
    'tenure': [3, 4, 5],
    'balance': [60000, 70000, 80000],
    'products_number': [2, 1, 3],
    'estimated_salary': [50000, 60000, 70000]
})

# 전처리된 데이터의 열 수 확인
imputer = SimpleImputer(strategy='mean')
dummy_customers[numeric_features] = imputer.fit_transform(dummy_customers[numeric_features])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

preprocessed_data = preprocessor.fit_transform(dummy_customers)
preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
input_dim = preprocessed_df.shape[1]
print(f'Input dimension: {input_dim}')  # 전처리된 데이터의 열 수 확인

# 각 범주형 특성의 고유값 수 확인
for feature in categorical_features:
    print(f'{feature}: {dummy_customers[feature].nunique()} unique values')

# 모델 인스턴스 생성 및 로드
model = ChurnModel(input_dim=input_dim).to(device)
model.load_state_dict(torch.load("churn_model.pth"))
model.eval()

# 예측 실행
for i, customer in dummy_customers.iterrows():
    customer_df = pd.DataFrame([customer])
    churn_probability = predict_customer_churn(model, customer_df, numeric_features, categorical_features, preprocessor)
    print(f'Customer {i} Churn Probability: {churn_probability:.4f}')