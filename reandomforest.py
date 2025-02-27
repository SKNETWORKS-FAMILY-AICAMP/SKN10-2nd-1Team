import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 랜덤 시드 고정
random_state = 42

# CSV 파일 불러오기 및 불필요한 컬럼 제거
df = pd.read_csv('data\Bank Customer Churn Prediction.csv')
if 'customer_id' in df.columns:
    df.drop('customer_id', axis=1, inplace=True)

# 입력과 타겟 분리 + 간단한 범주형 인코딩 (get_dummies 사용)
X = pd.get_dummies(df.drop('churn', axis=1), drop_first=True)
y = df['churn']

# 데이터 분할 (train, validation, test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_state)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

# SMOTE로 오버샘플링 수행 (train set에만 적용)
smote = SMOTE(random_state=random_state)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# random_state=42로 설정된 Random Forest 모델 정의 (n_estimators=100)
model = RandomForestClassifier(random_state=random_state, n_estimators=100)

# 모델 학습
model.fit(X_train_smote, y_train_smote)

# 검증 데이터로 예측
y_val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
val_loss = log_loss(y_val, y_val_pred)
val_accuracy = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))

print(f'Validation AUC: {val_auc:.2f}')
print(f'Validation Loss: {val_loss:.2f}')
print(f'Validation Accuracy: {val_accuracy:.2f}')

# 테스트 데이터로 예측
y_test_pred = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_pred)
test_loss = log_loss(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))

print(f'Test AUC: {test_auc:.2f}')
print(f'Test Loss: {test_loss:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')