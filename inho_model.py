import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset

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
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 모델 로드 함수
def load_model(filepath):
    model = ChurnModel()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

# 예측 함수
def predict(model, data):
    X = data[['credit_score', 'gender', 'age', 'tenure', 'balance',
              'products_number', 'credit_card', 'active_member', 'estimated_salary',
              'country_France', 'country_Germany', 'country_Spain']]
    X_tensor = torch.tensor(X.values).float()
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = outputs.squeeze().numpy()
        predictions = (probabilities > 0.5).astype(int)
    return predictions, probabilities