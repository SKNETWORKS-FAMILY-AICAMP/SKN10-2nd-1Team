import pandas as pd
import numpy as np
import joblib
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

def train_and_save_model():
    # CSV 파일 불러오기 및 불필요한 컬럼 제거
    df = pd.read_csv('data/Bank Customer Churn Prediction.csv')
    if 'customer_id' in df.columns:
        df.drop('customer_id', axis=1, inplace=True)

    # 입력과 타겟 분리 및 범주형 변수 인코딩 (get_dummies 사용)
    X = pd.get_dummies(df.drop('churn', axis=1), drop_first=True)
    y = df['churn']

    # 데이터 분할 (70% 학습, 20% 검증, 10% 테스트)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
    )

    # 전처리기 설정: 모든 피처에 대해 StandardScaler 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns)
        ]
    )

    # Stacking Classifier 구성 (기본 모델: LR, DT, RF, GB; 최종 모델: LR)
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

    # 파이프라인 구성
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stack_model)
    ])

    # 모델 학습 (학습 셋 사용)
    print("모델 학습 중...")
    pipeline.fit(X_train, y_train)

    # 검증 셋 성능 평가
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"검증 셋 정확도: {val_accuracy:.4f}")

    # 테스트 셋 성능 평가
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"테스트 셋 정확도: {test_accuracy:.4f}")

    # 모델 저장
    print("모델 저장 중...")
    joblib.dump(pipeline, 'churn_prediction_model.joblib')
    print("모델이 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    train_and_save_model()
