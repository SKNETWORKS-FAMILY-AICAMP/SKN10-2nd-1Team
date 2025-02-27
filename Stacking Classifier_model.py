import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# CSV 파일 불러오기 및 불필요한 컬럼 제거
df = pd.read_csv('Bank Customer Churn Prediction.csv')
if 'customer_id' in df.columns:
    df.drop('customer_id', axis=1, inplace=True)

# 입력과 타겟 분리 및 범주형 변수 인코딩 (get_dummies 사용)
X = pd.get_dummies(df.drop('churn', axis=1), drop_first=True)
y = df['churn']

# 데이터 분할 (학습: 80%, 테스트: 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 모든 피처는 get_dummies 결과이므로 모두 수치형이 됨 → 전체 피처에 대해 스케일링 적용
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)
    ]
)

# Stacking Classifier 구성 (기본 모델: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
base_estimators = [
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
    ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100))
]
stack_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# 전처리와 Stacking Classifier를 하나의 파이프라인에 구성
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stack_model)
])

# 모델 학습
pipeline.fit(X_train, y_train)

# 테스트 셋에 대한 예측 (클래스)
y_pred = pipeline.predict(X_test)
print("테스트 셋 예측 결과:")
print(y_pred)

# 테스트 셋에 대한 예측 확률 (클래스 1에 대한 확률)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print("테스트 셋 예측 확률 (클래스 1):")
print(y_proba)

# 테스트 정확도 출력
acc = accuracy_score(y_test, y_pred)
print(f"테스트 셋 정확도: {acc:.4f}")
