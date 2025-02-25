import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
import joblib

# 한글 폰트 설정 (Windows: 'Malgun Gothic', Mac: 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def eda_visualizations(df):
    # 1. 고객 이탈 분포
    plt.figure(figsize=(6,4))
    sns.countplot(x='churn', data=df, palette='viridis')
    plt.title('고객 이탈 분포')
    plt.xlabel('Churn (0: 유지, 1: 이탈)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # 2. 수치형 변수 히스토그램 및 박스플롯
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('churn')
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'{col} 분포')
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'{col} 박스플롯')
        plt.tight_layout()
        plt.show()

    # 3. 상관관계 히트맵
    plt.figure(figsize=(8,6))
    corr = df[numeric_cols + ['churn']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('상관관계 히트맵')
    plt.tight_layout()
    plt.show()

def data_preprocessing(df):
    # 불필요한 변수 제거
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    X = df.drop('churn', axis=1)
    y = df['churn']
    categorical_features = ['country', 'gender']
    numeric_features = [col for col in X.columns if col not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    return X, y, preprocessor, numeric_features, categorical_features

def split_data(X, y):
    # 70% train, 20% validation, 10% test (stratify 사용)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{model_name}] 테스트 정확도: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} 혼동 행렬')
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.tight_layout()
    plt.show()
    
    # ROC 커브 및 AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC 커브')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        # Precision-Recall 커브
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall 커브')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    return acc

def main():
    # CSV 파일 불러오기
    df = pd.read_csv('Bank Customer Churn Prediction.csv')
    
    # 심화 EDA 수행
    eda_visualizations(df)
    
    # 데이터 전처리
    X, y, preprocessor, numeric_features, categorical_features = data_preprocessing(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 전처리 파이프라인 적용
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)
    
    # OneHotEncoder 후 피처 이름 추출
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    ohe_features = ohe.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_features)
    
    models = {}
    results = {}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Logistic Regression (GridSearchCV 적용)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_param_grid = {'C': [0.1, 1, 10]}
    lr_grid = GridSearchCV(lr, lr_param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train_proc, y_train)
    best_lr = lr_grid.best_estimator_
    print("Logistic Regression 최적 파라미터:", lr_grid.best_params_)
    models['Logistic Regression'] = best_lr
    results['Logistic Regression'] = evaluate_model(best_lr, X_test_proc, y_test, "Logistic Regression")
    
    # 2. Decision Tree (GridSearchCV 적용)
    dt = DecisionTreeClassifier(random_state=42)
    dt_param_grid = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}
    dt_grid = GridSearchCV(dt, dt_param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    dt_grid.fit(X_train_proc, y_train)
    best_dt = dt_grid.best_estimator_
    print("Decision Tree 최적 파라미터:", dt_grid.best_params_)
    models['Decision Tree'] = best_dt
    results['Decision Tree'] = evaluate_model(best_dt, X_test_proc, y_test, "Decision Tree")
    
    # 3. Random Forest (GridSearchCV 적용)
    rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train_proc, y_train)
    best_rf = rf_grid.best_estimator_
    print("Random Forest 최적 파라미터:", rf_grid.best_params_)
    models['Random Forest'] = best_rf
    results['Random Forest'] = evaluate_model(best_rf, X_test_proc, y_test, "Random Forest")
    
    # 4. Gradient Boosting (GridSearchCV 적용)
    gb = GradientBoostingClassifier(random_state=42)
    gb_param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    gb_grid = GridSearchCV(gb, gb_param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train_proc, y_train)
    best_gb = gb_grid.best_estimator_
    print("Gradient Boosting 최적 파라미터:", gb_grid.best_params_)
    models['Gradient Boosting'] = best_gb
    results['Gradient Boosting'] = evaluate_model(best_gb, X_test_proc, y_test, "Gradient Boosting")
    
    # 5. Stacking Classifier (앙상블 모델)
    estimators = [
        ('lr', best_lr),
        ('dt', best_dt),
        ('rf', best_rf),
        ('gb', best_gb)
    ]
    stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=cv_strategy, n_jobs=-1)
    stack_model.fit(X_train_proc, y_train)
    models['Stacking Classifier'] = stack_model
    results['Stacking Classifier'] = evaluate_model(stack_model, X_test_proc, y_test, "Stacking Classifier")
    
    # 모델별 정확도 비교 바 차트
    plt.figure(figsize=(8,5))
    model_names = list(results.keys())
    accuracies = list(results.values())
    sns.barplot(x=model_names, y=accuracies, palette='muted')
    plt.ylim(0, 1)
    plt.ylabel('테스트 정확도')
    plt.title('모델별 테스트 정확도 비교')
    for i, acc in enumerate(accuracies):
        plt.text(i, acc+0.02, f"{acc:.2f}", ha='center')
    plt.tight_layout()
    plt.show()

    # 트리 기반 모델의 피처 중요도 시각화 (Random Forest, Gradient Boosting)
    for model_name in ['Random Forest', 'Gradient Boosting']:
        model = models[model_name]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            plt.figure(figsize=(10,6))
            sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
            plt.title(f'{model_name} 피처 중요도')
            plt.xlabel('중요도')
            plt.ylabel('피처')
            plt.tight_layout()
            plt.show()
    
    # 최종 모델(여기서는 스태킹 분류기)을 파일로 저장
    joblib.dump(models['Stacking Classifier'], 'best_churn_model.pkl')
    print("최종 모델이 'best_churn_model.pkl'로 저장되었습니다.")

if __name__ == '__main__':
    main()
