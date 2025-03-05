import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 랜덤 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# 데이터 전처리
def preprocess_data(data):
    # 범주형과 수치형 특성 정의
    categorical_features = ['country', 'gender', 'credit_card', 'active_member']
    numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

    # 결측값 처리
    imputer = SimpleImputer(strategy='mean')
    data[numeric_features] = imputer.fit_transform(data[numeric_features])

    # 전처리
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    preprocessed_data = preprocessor.fit_transform(data)
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

    return preprocessed_df

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

# 모델 로드 함수
def load_model(filepath, input_dim):
    model = ChurnModel(input_dim=input_dim)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

# 예측 함수
def predict(model, data, preprocessor, numeric_features, categorical_features):
    preprocessed_data = preprocessor.transform(data)
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    X_tensor = torch.tensor(preprocessed_df.values).float().to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = outputs.squeeze().cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)
    return predictions, probabilities