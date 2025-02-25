import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_submit(test_df, features, model):
    today = datetime.today().strftime('%Y-%m-%d')
    submit_df = pd.read_csv('./data/sample_submission.csv')

    submit_df['채무 불이행 확률'] = model.predict(test_df[features])
    submit_df.to_csv(f'./data/submission_{today}.csv', index=False)



def base_model(X, y, best_params=None):
    # 데이터 분할 (입력 변수와 목표 변수)

    # 데이터 분할 (학습 데이터와 테스트 데이터)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    if best_params is None:
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)

    return model

def feature_importance(model, X):
    # 피처 중요도 계산
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 중요도에 따라 정렬

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()
