import pandas as pd
import joblib
import pickle
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from module.inho_model import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_churn(filtered_data, model_select:str, df=pd.read_csv('./data/Bank Customer Churn Prediction.csv')):
    if model_select == 'Gradient Boosting (AUC: 0.8585)':
        # customer_id 컬럼이 있다면 제거
        if 'customer_id' in filtered_data.columns:
            filtered_data = filtered_data.drop('customer_id', axis=1)
        
        # churn 컬럼이 있다면 제거
        if 'churn' in filtered_data.columns:
            filtered_data = filtered_data.drop('churn', axis=1)
        
        # 원-핫 인코딩 적용
        X_new = pd.get_dummies(filtered_data, drop_first=True)
        
        # 저장된 모델 불러오기
        try:
            pipeline = joblib.load('./model/churn_prediction_model.joblib')
        except FileNotFoundError:
            raise Exception("모델 파일을 찾을 수 없습니다. 먼저 모델을 학습하고 저장해주세요.")
        
        # 예측 수행
        predictions = pipeline.predict(X_new)
        probabilities = pipeline.predict_proba(X_new)[:, 1]

        return predictions, probabilities
    
    elif model_select == 'Random Forest (AUC: 0.8589)':
        # 전처리
        filtered_data['country_France'] = filtered_data['country'].apply(lambda x: 1 if x == 'France' else 0)
        filtered_data['country_Germany'] = filtered_data['country'].apply(lambda x: 1 if x == 'Germany' else 0)
        filtered_data['country_Spain'] = filtered_data['country'].apply(lambda x: 1 if x == 'Spain' else 0)

        filtered_data['gender'] = filtered_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)

        pt = PowerTransformer(method='yeo-johnson')
        pt.fit_transform(df['credit_score'].values.reshape(-1, 1))
        df['credit_score'] = pt.transform(df['credit_score'].values.reshape(-1,1))
        filtered_data['credit_score'] = pt.transform(filtered_data['credit_score'].values.reshape(-1,1))
        pt.fit_transform(df['age'].values.reshape(-1, 1))
        df['age'] = pt.transform(df['age'].values.reshape(-1,1))
        filtered_data['age'] = pt.transform(filtered_data['age'].values.reshape(-1,1))

        scaler = StandardScaler()
        scaler.fit_transform(df['credit_score'].values.reshape(-1, 1))
        filtered_data['credit_score'] = scaler.transform(filtered_data['credit_score'].values.reshape(-1,1))
        scaler.fit_transform(df['age'].values.reshape(-1, 1))
        filtered_data['age'] = scaler.transform(filtered_data['age'].values.reshape(-1,1))
        scaler.fit_transform(df['balance'].values.reshape(-1, 1))
        filtered_data['balance'] = scaler.transform(filtered_data['balance'].values.reshape(-1,1))
        scaler.fit_transform(df['estimated_salary'].values.reshape(-1, 1))
        filtered_data['estimated_salary'] = scaler.transform(filtered_data['estimated_salary'].values.reshape(-1,1))

        # 모델 불러오기
        model = pickle.load(open("./model/randomforest_model.pkl", "rb"))

        # 예측
        X = filtered_data[['credit_score', 'gender', 'age', 'tenure', 'balance',
                           'products_number', 'credit_card', 'active_member', 'estimated_salary',
                           'country_France', 'country_Germany', 'country_Spain']]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        return predictions, probabilities

    if model_select == 'Deep Learning (AUC: 0.8612)':
        # 범주형과 수치형 특성 정의
        categorical_features = ['country', 'gender', 'credit_card', 'active_member']
        numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

        # 전처리된 데이터의 열 수 확인
        imputer = SimpleImputer(strategy='mean')
        df[numeric_features] = imputer.fit_transform(df[numeric_features])
        filtered_data[numeric_features] = imputer.transform(filtered_data[numeric_features])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        preprocessed_data = preprocessor.fit_transform(df)
        preprocessed_filtered_data = preprocessor.transform(filtered_data)
        preprocessed_df = pd.DataFrame(preprocessed_filtered_data, columns=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
        input_dim = preprocessed_df.shape[1]
        print(f'Input dimension: {input_dim}')  # 전처리된 데이터의 열 수 확인

        # 모델 불러오기
        model = load_model('model/churn_model_DL.pth', input_dim=input_dim)
        model.eval()

        # 예측 수행
        X_tensor = torch.tensor(preprocessed_df.values).float().to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = outputs.squeeze().cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)

        return predictions, probabilities