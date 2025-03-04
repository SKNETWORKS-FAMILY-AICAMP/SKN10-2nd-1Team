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
    data = data.drop(['customer_id'], axis=1)
    data['country_France'] = data['country'].apply(lambda x: 1 if x == 'France' else 0)
    data['country_Germany'] = data['country'].apply(lambda x: 1 if x == 'Germany' else 0)
    data['country_Spain'] = data['country'].apply(lambda x: 1 if x == 'Spain' else 0)
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    pt = PowerTransformer(method='yeo-johnson')
    data['credit_score'] = pt.fit_transform(data['credit_score'].values.reshape(-1, 1))
    data['age'] = pt.fit_transform(data['age'].values.reshape(-1, 1))

    scaler = StandardScaler()
    data['credit_score'] = scaler.fit_transform(data['credit_score'].values.reshape(-1, 1))
    data['age'] = scaler.fit_transform(data['age'].values.reshape(-1, 1))
    data['balance'] = scaler.fit_transform(data['balance'].values.reshape(-1, 1))
    data['estimated_salary'] = scaler.fit_transform(data['estimated_salary'].values.reshape(-1, 1))

    return data

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