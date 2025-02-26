import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")

# CSV 파일 불러오기 및 불필요한 컬럼 제거
df = pd.read_csv('Bank Customer Churn Prediction.csv')
if 'customer_id' in df.columns:
    df.drop('customer_id', axis=1, inplace=True)

# 입력과 타겟 분리 + 간단한 범주형 인코딩 (get_dummies 사용)
X = pd.get_dummies(df.drop('churn', axis=1), drop_first=True)
y = df['churn']

# SMOTE로 오버샘플링 수행 (random_state=42)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# random_state=42로 설정된 Random Forest 모델 정의 (n_estimators=100)
model = RandomForestClassifier(random_state=42, n_estimators=100)

# 3-폴드 교차 검증으로 평가 지표 계산
cv_results = cross_validate(model, X_smote, y_smote, cv=3,
                            scoring=["accuracy", "f1", "roc_auc"],
                            n_jobs=-1)

# 평균 점수 계산
f1 = cv_results['test_f1'].mean()
auc_score = cv_results['test_roc_auc'].mean()
accuracy = cv_results['test_accuracy'].mean()

print(f'f1: {f1:.2f}')
print(f'auc: {auc_score:.2f}')
print(f'accuracy: {accuracy:.2f}')
