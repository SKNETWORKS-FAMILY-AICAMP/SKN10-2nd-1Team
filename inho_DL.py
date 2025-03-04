import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# 랜덤 시드 고정
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
def load_data():
    data = pd.read_csv('data/Bank Customer Churn Prediction.csv')
    return data

# 불필요한 특성 제거
data = load_data()
data = data.drop(['customer_id'], axis=1)

# 범주형과 수치형 특성 분리
categorical_features = ['country', 'gender']
numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

def preprocess_data(data, numeric_features, categorical_features):
    # 결측치 처리
    imputer = SimpleImputer(strategy='mean')
    data[numeric_features] = imputer.fit_transform(data[numeric_features])

    # 전처리기를 구성: 수치형 데이터에는 StandardScaler, 범주형 데이터에는 OneHotEncoder 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # 데이터 전처리
    preprocessed_data = preprocessor.fit_transform(data)
    
    # 전처리된 데이터를 DataFrame으로 변환
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    preprocessed_df.reset_index(drop=True, inplace=True)
    return preprocessed_df, data

# 데이터 전처리
preprocessed_data, processed_data = preprocess_data(data, numeric_features, categorical_features)

# 데이터 분할 (7:2:1 비율)
X = preprocessed_data
y = processed_data['churn']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SMOTE 적용
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# PyTorch 데이터셋과 데이터 로더
train_dataset = TensorDataset(torch.tensor(X_train_smote.values).float(), torch.tensor(y_train_smote.values).float())
val_dataset = TensorDataset(torch.tensor(X_val.values).float(), torch.tensor(y_val.values).float())
test_dataset = TensorDataset(torch.tensor(X_test.values).float(), torch.tensor(y_test.values).float())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 모델 정의
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(15, 256),
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

# 모델 인스턴스 생성 및 설정
model = ChurnModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# 얼리 스토퍼 및 모델 학습 함수
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, val_auc):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    train_losses = []
    val_aucs = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_labels = []
        train_outputs = []
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_labels.extend(labels.tolist())
            train_outputs.extend(outputs.squeeze().tolist())

        train_accuracy = accuracy_score(train_labels, [1 if o > 0.5 else 0 for o in train_outputs])
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_labels = []
        val_outputs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_labels.extend(labels.tolist())
                val_outputs.extend(outputs.squeeze().tolist())

        val_auc = roc_auc_score(val_labels, val_outputs)
        val_loss = total_loss / len(train_loader)
        val_accuracy = accuracy_score(val_labels, [1 if o > 0.5 else 0 for o in val_outputs])
        train_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch+1}, Training Loss: {val_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation AUC: {val_auc:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        early_stopping(val_loss, val_auc)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step(val_loss)

    # Plotting the results
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_aucs, 'b', label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# 학습 실행
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50)

# 테스트 데이터로 예측
model.eval()
test_labels = []
test_outputs = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_labels.extend(labels.tolist())
        test_outputs.extend(outputs.squeeze().tolist())

test_auc = roc_auc_score(test_labels, test_outputs)
test_loss = log_loss(test_labels, test_outputs)
test_accuracy = accuracy_score(test_labels, [1 if o > 0.5 else 0 for o in test_outputs])

print(f'Test AUC: {test_auc:.4f}')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
