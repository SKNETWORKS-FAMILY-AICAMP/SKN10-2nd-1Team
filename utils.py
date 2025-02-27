import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
import random
import os

def reset_seeds(func, seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 파이썬 환경변수 시드 고정
    np.random.seed(seed)

    def wrapper_func(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_func

@reset_seeds
def hpo(X_train, y_train):
    
    params = {
        'n_estimators': randint(10, 300),
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
    }

    rf = RandomForestClassifier(random_state=42)
    random_search_cv = RandomizedSearchCV(rf, param_distributions=params, cv=3, n_jobs=-1, random_state=42, n_iter=100, scoring='roc_auc')
    random_search_cv.fit(X_train, y_train)

    print('최적 하이퍼 파라미터: ', random_search_cv.best_params_)
    print('최고 예측 정확도: {:.4f}'.format(random_search_cv.best_score_))

    return random_search_cv.best_params_


@reset_seeds
def base_model(X, y):

    # 데이터 분할 (학습 데이터와 테스트 데이터)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 불균형 데이터 처리
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 모델 학습
    best_params = hpo(X_train, y_train)
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # 모델 평가
    model_report(model, X_test, y_test)

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


def preprocessing(df):
    # 범주형 데이터 인코딩
    df = pd.get_dummies(df, columns=['country'])
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # products_number 2 이상 데이터 처리
    # df['products_number'] = df['products_number'].apply(lambda x: 3 if x >= 3 else x)

    # 수치형 데이터 범주화    
    #df['credit_score_seg'] = df['credit_score'].apply(lambda x: 0 if x <= 349 else (1 if x <= 500 else (2 if x <= 590 else (3 if x <= 620 else (4 if x <= 660 else (5 if x <= 690 else (6 if x <= 720 else 7)))))))
    #df['balance_seg'] = df['balance'].apply(lambda x: 0 if x <= 50000 else (1 if x <= 90000 else (2 if x <= 127000 else 3)))
    #df['age_seg'] = df['age'].apply(lambda x: 0 if x <= 17 else (1 if x <= 36 else (2 if x <= 55 else 3)))
    #df['estimated_salary_seg'] = df['estimated_salary'].apply(lambda x: 0 if x <= 51002 else (1 if x <= 100193 else (2 if x <= 149388 else 3)))
    
    # balance 0 값 처리  (balance_0 컬럼 추가 후 0값을 중앙값으로 대체)
    #df['balance_0'] = df['balance'].apply(lambda x: 1 if x == 0 else 0)
    #median_balance = df.loc[df['balance'] != 0, 'balance'].median()
    #df.loc[df['balance'] == 0, 'balance'] = median_balance

    # PowerTransformer를 사용한 데이터 변환
    pt = PowerTransformer(method='yeo-johnson')
    df['credit_score'] = pt.fit_transform(df['credit_score'].values.reshape(-1, 1))
    df['age'] = pt.fit_transform(df['age'].values.reshape(-1, 1))
    #df['balance'] = pt.fit_transform(df['balance'].values.reshape(-1, 1))
    #df['estimated_salary'] = pt.fit_transform(df['estimated_salary'].values.reshape(-1, 1))

    # 'credit_score', 'age', 'balance', 'estimated_salary' 데이터 스케일링
    scale_columns = ['credit_score', 'age', 'balance', 'estimated_salary', ]
    scaler = StandardScaler()
    df = pd.concat([df.drop(scale_columns, axis=1), pd.DataFrame(scaler.fit_transform(df[scale_columns]), columns=scale_columns)], axis=1)

    df_target = df['churn']
    df_feature = df.drop(['churn', 'customer_id',], axis=1)
    return df_feature, df_target

def roc_auc_curve_plt(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def model_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    print('Classification Report:')
    print(report)

    roc_auc_curve_plt(y_test, y_pred_proba)


