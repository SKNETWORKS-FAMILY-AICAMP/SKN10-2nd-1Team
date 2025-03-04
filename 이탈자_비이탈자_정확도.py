import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from inho_model import load_model

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
data = pd.read_csv('data/Bank Customer Churn Prediction.csv')

# 범주형과 수치형 특성 정의
categorical_features = ['country', 'gender', 'credit_card', 'active_member']
numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

# 전처리
imputer = SimpleImputer(strategy='mean')
data[numeric_features] = imputer.fit_transform(data[numeric_features])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

preprocessed_data = preprocessor.fit_transform(data)
preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

# 모델 불러오기
input_dim = preprocessed_df.shape[1]
model = load_model('model/churn_model_DL.pth', input_dim=input_dim)
model.eval()

# 예측 수행
X_tensor = torch.tensor(preprocessed_df.values).float().to(device)
with torch.no_grad():
    outputs = model(X_tensor)
    probabilities = outputs.squeeze().cpu().numpy()
    predictions = (probabilities > 0.5).astype(int)

# 실제 값과 예측 값 비교
actuals = data['churn'].values
accuracy = accuracy_score(actuals, predictions)
print(f'전체 데이터의 예측 정확도: {accuracy:.4f}')

churn_indices = actuals == 1
non_churn_indices = actuals == 0
# 이탈자와 비이탈자 총수
total_churn = churn_indices.sum()
total_non_churn = non_churn_indices.sum()
print(f'이탈자 총수: {total_churn}')
print(f'비이탈자 총수: {total_non_churn}')
# 이탈자 예측 정확도
churn_accuracy = accuracy_score(actuals[churn_indices], predictions[churn_indices])
print(f'이탈자 예측 정확도: {churn_accuracy:.4f}')

# 비이탈자 예측 정확도
non_churn_accuracy = accuracy_score(actuals[non_churn_indices], predictions[non_churn_indices])
print(f'비이탈자 예측 정확도: {non_churn_accuracy:.4f}')


# 이탈자 중에 이탈로 판단한 숫자
correct_churn_predictions = (predictions[churn_indices] == 1).sum()
print(f'이탈자 중에 이탈로 판단한 숫자: {correct_churn_predictions}')

# 비이탈자 중에 비이탈로 판단한 숫자
correct_non_churn_predictions = (predictions[non_churn_indices] == 0).sum()
print(f'비이탈자 중에 비이탈로 판단한 숫자: {correct_non_churn_predictions}')