import streamlit as st
import pandas as pd
import joblib
import warnings
import pickle
import torch
from groq import Groq
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from inho_model import load_data, preprocess_data, load_model, predict, ChurnModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# Groq API 키 설정
GROQ_API_KEY = "너의 api키"

# Groq 클라이언트 초기화
client = Groq(api_key=GROQ_API_KEY)

# 스타일 설정
st.set_page_config(page_title="은행 고객 이탈 예측", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .stSlider {
        color: #0073e6;
    }
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .low-risk {
        color: #00cc00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

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

def generate_churn_analysis_data(results_df):
    """ 모델 예측 결과 기반으로 위험도별 주요 분석 데이터를 생성 """
    
    results_df["risk_level"] = pd.cut(results_df["이탈 확률"], bins=[0, 0.4, 0.7, 1.0], labels=["낮음", "중간", "높음"])
    risk_counts = results_df["risk_level"].value_counts().to_dict()
    risk_group_means = results_df.groupby("risk_level")[["credit_score", "balance", "estimated_salary"]].mean().to_dict()

    results_df["age_group"] = pd.cut(results_df["age"], bins=[18, 30, 40, 50, 60, 100], labels=["20대", "30대", "40대", "50대", "60대 이상"])
    age_churn_rates = results_df.groupby("age_group")["이탈 확률"].mean() * 100
    country_churn_rates = results_df.groupby("country")["이탈 확률"].mean() * 100
    gender_churn_rates = results_df.groupby("gender")["이탈 확률"].mean() * 100

    return {
        "risk_counts": risk_counts,
        "risk_group_means": risk_group_means,
        "age_churn_rates": age_churn_rates.to_dict(),
        "country_churn_rates": country_churn_rates.to_dict(),
        "gender_churn_rates": gender_churn_rates.to_dict()
    }

def generate_prompt_from_analysis(analysis_data):
    """ 분석된 데이터를 바탕으로 Groq API 요청을 위한 프롬프트 생성 """
    
    prompt = f"""
    ### 고객 이탈 분석 요청 (한국어로 작성)
    주어진 데이터를 바탕으로 고객 이탈 원인과 해결 방안을 도출하시오.
    # 돈단위는 유로(€)로 표기합니다.
    ### 🔹 기본 정보
    - 총 고객 수: {sum(analysis_data["risk_counts"].values())}명
    - 높은 위험 고객 수: {analysis_data["risk_counts"].get("높음", 0)}명
    - 중간 위험 고객 수: {analysis_data["risk_counts"].get("중간", 0)}명
    - 낮은 위험 고객 수: {analysis_data["risk_counts"].get("낮음", 0)}명

    ### 🔹 위험 수준별 고객 특성
    📌 **높은 위험 고객**
    - 평균 신용 점수: {analysis_data["risk_group_means"]["credit_score"].get("높음", "N/A")}
    - 평균 계좌 잔액: {analysis_data["risk_group_means"]["balance"].get("높음", "N/A")}
    - 평균 연봉: {analysis_data["risk_group_means"]["estimated_salary"].get("높음", "N/A")}

    📌 **중간 위험 고객**
    - 평균 신용 점수: {analysis_data["risk_group_means"]["credit_score"].get("중간", "N/A")}
    - 평균 계좌 잔액: {analysis_data["risk_group_means"]["balance"].get("중간", "N/A")}
    - 평균 연봉: {analysis_data["risk_group_means"]["estimated_salary"].get("중간", "N/A")}

    📌 **낮은 위험 고객**
    - 평균 신용 점수: {analysis_data["risk_group_means"]["credit_score"].get("낮음", "N/A")}
    - 평균 계좌 잔액: {analysis_data["risk_group_means"]["balance"].get("낮음", "N/A")}
    - 평균 연봉: {analysis_data["risk_group_means"]["estimated_salary"].get("낮음", "N/A")}

    ### 🔹 연령대별, 국가별, 성별 이탈률
    📌 **연령대별 이탈률 (%)**
    {analysis_data["age_churn_rates"]}

    📌 **국가별 이탈률 (%)**
    {analysis_data["country_churn_rates"]}

    📌 **성별 이탈률 (%)**
    {analysis_data["gender_churn_rates"]}


    ### 응답 형식 (항상 이 형식 유지)
    원인이 수치적 데이터와 관련있다면 수치적으로 분석할것

    - 원인 1
        - 설명
        - 해결방안
    - 원인 2
        - 설명
        - 해결방안
    - 원인 3
        - 설명
        - 해결방안

    """


    return prompt

def main():
    st.title('은행 고객 이탈 예측 시스템')
    
    # 데이터 로드
    df = load_data('./data/Bank Customer Churn Prediction.csv')
    
    # 표시할 컬럼 설정
    display_columns = ['customer_id', 'country', 'age', 'balance', '이탈 예측', '이탈 확률']
    
    # 세션 상태 초기화
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    # 사이드바에 필터 옵션 배치
    st.sidebar.header('필터 옵션')
    
    # 수치형 변수 필터 (사이드바)
    credit_score = st.sidebar.slider(
        '신용점수',
        int(df['credit_score'].min()),
        int(df['credit_score'].max()),
        (int(df['credit_score'].min()), int(df['credit_score'].max()))
    )
    
    age = st.sidebar.slider(
        '나이',
        int(df['age'].min()),
        int(df['age'].max()),
        (int(df['age'].min()), int(df['age'].max()))
    )
    
    tenure = st.sidebar.slider(
        '거래기간',
        int(df['tenure'].min()),
        int(df['tenure'].max()),
        (int(df['tenure'].min()), int(df['tenure'].max()))
    )
    
    balance = st.sidebar.slider(
        '계좌잔액',
        float(df['balance'].min()),
        float(df['balance'].max()),
        (float(df['balance'].min()), float(df['balance'].max()))
    )
    
    country = st.sidebar.multiselect(
        '국가',
        df['country'].unique().tolist(),
        default=df['country'].unique().tolist()
    )
    
    gender = st.sidebar.multiselect(
        '성별',
        df['gender'].unique().tolist(),
        default=df['gender'].unique().tolist()
    )
    
    products_number = st.sidebar.multiselect(
        '상품 수',
        df['products_number'].unique().tolist(),
        default=df['products_number'].unique().tolist()
    )
    
    credit_card = st.sidebar.multiselect(
        '신용카드 보유',
        [0, 1],
        default=[0, 1]
    )
    
    active_member = st.sidebar.multiselect(
        '활성 회원',
        [0, 1],
        default=[0, 1]
    )
    churn = st.sidebar.multiselect(
        '이탈 여부',
        [0, 1],
        default=[0, 1]
    )
    # 데이터 필터링
    filtered_df = df[
        (df['credit_score'].between(credit_score[0], credit_score[1])) &
        (df['age'].between(age[0], age[1])) &
        (df['tenure'].between(tenure[0], tenure[1])) &
        (df['balance'].between(balance[0], balance[1])) &
        (df['country'].isin(country)) &
        (df['gender'].isin(gender)) &
        (df['products_number'].isin(products_number)) &
        (df['credit_card'].isin(credit_card)) &
        (df['active_member'].isin(active_member)) &
        (df['churn'].isin(churn))
    ]
    
    # 필터링된 데이터 표시
    st.write(f"필터링된 고객 수: {len(filtered_df):,}명")
    st.dataframe(filtered_df)
    
    accuracy_dict = {
        'Gradient Boosting': 0.8730,
        'Random Forest': 0.8340,
        'Deep Learning': 0.8640
    }
    auc_dict = {
        'Gradient Boosting': 0.8633,
        'Random Forest': 0.8589,
        'Deep Learning': 0.8612
    }

    # 예측 버튼과 모델 선택 박스
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        model_select = st.selectbox('모델 선택', ['Gradient Boosting (AUC: 0.8585)', 'Random Forest (AUC: 0.8589)', 'Deep Learning (AUC: 0.8612)'], index=0)
        if st.button('이탈 예측하기', use_container_width=True):
            if len(filtered_df) > 0:
                with st.spinner('예측 중...'):
                    results_df = filtered_df.copy()
                    predictions, probabilities = predict_churn(filtered_df, model_select)
                    
                    # 결과를 데이터프레임에 추가
                    
                    results_df['이탈 예측'] = ['이탈 예정' if p == 1 else '유지 예정' for p in predictions]
                    results_df['이탈 확률'] = probabilities
                    
                    # 결과를 세션 상태에 저장
                    st.session_state.results_df = results_df
                    
                    # 결과 표시
                    st.success('예측이 완료되었습니다!')
                    
                    # 통계 지표
                    total_customers = len(results_df)
                    predicted_churns = sum(predictions)
                    churn_rate = (predicted_churns / total_customers) * 100
                    
                    # 위험도별 고객 수 계산
                    high_risk = len(results_df[results_df['이탈 확률'] >= 0.7])
                    medium_risk = len(results_df[(results_df['이탈 확률'] >= 0.4) & (results_df['이탈 확률'] < 0.7)])
                    low_risk = len(results_df[results_df['이탈 확률'] < 0.4])
                    
                    # 전체 통계를 한 줄로 표시
                    st.markdown("### 예측 결과 요약")
                    
                    # CSS로 메트릭 스타일 조정
                    st.markdown("""
                    <style>
                    [data-testid="stMetricValue"] {
                        font-size: 24px;
                    }
                    [data-testid="stMetricDelta"] {
                        font-size: 16px;
                    }
                    [data-testid="stMetricLabel"] {
                        font-size: 16px;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # 메트릭을 5개의 동일한 크기 컬럼으로 나누기
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric(
                            "전체 고객",
                            f"{total_customers:,}명"
                        )
                    
                    with col2:
                        st.metric(
                            "이탈 예정",
                            f"{predicted_churns:,}명",
                            f"{churn_rate:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "높은 위험",
                            f"{high_risk:,}명",
                            f"{(high_risk/total_customers)*100:.1f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "중간 위험",
                            f"{medium_risk:,}명",
                            f"{(medium_risk/total_customers)*100:.1f}%"
                        )
                    
                    with col5:
                        st.metric(
                            "낮은 위험",
                            f"{low_risk:,}명",
                            f"{(low_risk/total_customers)*100:.1f}%"
                        )

                    # 구분선 추가
                    st.markdown("---")
                    
                    # 위험도별 고객 목록
                    st.markdown("### 위험도별 고객 목록")
                    
                    tab1, tab2, tab3 = st.tabs(["🔴 높은 위험", "🟡 중간 위험", "🟢 낮은 위험"])
                    
                    def style_dataframe(df):
                        def highlight_risk(val):
                            try:
                                prob = float(val.strip('%')) / 100
                                if prob >= 0.7:
                                    return 'background-color: #ffcccc'
                                elif prob >= 0.4:
                                    return 'background-color: #fff2cc'
                                else:
                                    return 'background-color: #d9ead3'
                            except:
                                return ''
                        
                        return df.style.apply(lambda x: [''] * len(x) if x.name != '이탈 확률' 
                                            else [highlight_risk(v) for v in x], axis=0)\
                                    .set_properties(**{
                                        'text-align': 'left',
                                        'white-space': 'pre-wrap',
                                        'font-size': '14px',
                                        'padding': '10px'
                                    })\
                                    .set_table_styles([
                                        {'selector': 'th',
                                         'props': [('font-size', '14px'),
                                                  ('text-align', 'left'),
                                                  ('padding', '10px'),
                                                  ('white-space', 'pre-wrap')]},
                                        {'selector': 'td',
                                         'props': [('min-width', '100px')]}
                                    ])
                    
                    with tab1:
                        high_risk_df = results_df[results_df['이탈 확률'] >= 0.7].copy()
                        if not high_risk_df.empty:
                            high_risk_df['이탈 확률'] = high_risk_df['이탈 확률'].apply(lambda x: f"{x:.1%}")
                            st.dataframe(style_dataframe(high_risk_df[display_columns].sort_values('이탈 확률', ascending=False)),
                                         height=400, use_container_width=True)
                        else:
                            st.info("높은 위험군에 해당하는 고객이 없습니다.")
                    
                    with tab2:
                        medium_risk_df = results_df[(results_df['이탈 확률'] >= 0.4) & 
                                                    (results_df['이탈 확률'] < 0.7)].copy()
                        if not medium_risk_df.empty:
                            medium_risk_df['이탈 확률'] = medium_risk_df['이탈 확률'].apply(lambda x: f"{x:.1%}")
                            st.dataframe(style_dataframe(medium_risk_df[display_columns].sort_values('이탈 확률', ascending=False)),
                                         height=400, use_container_width=True)
                        else:
                            st.info("중간 위험군에 해당하는 고객이 없습니다.")
                    
                    with tab3:
                        low_risk_df = results_df[results_df['이탈 확률'] < 0.4].copy()
                        if not low_risk_df.empty:
                            low_risk_df['이탈 확률'] = low_risk_df['이탈 확률'].apply(lambda x: f"{x:.1%}")
                            st.dataframe(style_dataframe(low_risk_df[display_columns].sort_values('이탈 확률', ascending=False)),
                                         height=400, use_container_width=True)
                        else:
                            st.info("낮은 위험군에 해당하는 고객이 없습니다.")

                    
                    st.markdown("""
                    ### 위험도 기준
                    - 🔴 높은 위험: 이탈 확률 70% 이상
                    - 🟡 중간 위험: 이탈 확률 40% ~ 70% 미만
                    - 🟢 낮은 위험: 이탈 확률 40% 미만
                    """)
                                        # 각 위험 수준에 속한 고객들의 정보 계산
                    def calculate_risk_info(df):
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                        
                        numeric_info = df[numeric_cols].mean().to_dict()
                        categorical_info = df[categorical_cols].apply(lambda x: x.value_counts().to_dict()).to_dict()
                        
                        return {**numeric_info, **categorical_info}
                    
                    high_risk_info = calculate_risk_info(high_risk_df)
                    medium_risk_info = calculate_risk_info(medium_risk_df)
                    low_risk_info = calculate_risk_info(low_risk_df)
                    
                    # 필터 정보 수집
                    filters = {
                        "신용점수": credit_score,
                        "나이": age,
                        "거래기간": tenure,
                        "계좌잔액": balance,
                        "국가": country,
                        "성별": gender,
                        "상품 수": products_number,
                        "신용카드 보유": credit_card,
                        "활성 회원": active_member
                    }
                    
                                        # 분석 데이터 생성
                    analysis_data = generate_churn_analysis_data(results_df)

                    # Groq API 프롬프트 생성
                    churn_analysis_prompt = generate_prompt_from_analysis(analysis_data)

                    # Groq API 요청
                    churn_reasons_solutions = client.chat.completions.create(
                        messages=[{"role": "user", "content": churn_analysis_prompt}],
                        model="qwen-2.5-coder-32b",
                    ).choices[0].message.content

                    # Streamlit에 표시
                    st.markdown("### 고객 이탈 원인 및 해결 방안")
                    st.markdown(churn_reasons_solutions)

                    # 위험도 기준 설명
                   

                    # 구분선 추가
                    st.markdown("---")
                    
                    # 데이터 다운로드 섹션
                    st.markdown("### 예측 결과 다운로드")
                    
                    # 다운로드 버튼들을 4개의 컬럼으로 배치
                    download_cols = st.columns(4)
                    
                    # 전체 데이터 다운로드
                    with download_cols[0]:
                        all_data = results_df[display_columns].copy()
                        all_data['이탈 확률'] = all_data['이탈 확률'].apply(lambda x: f"{x:.1%}")
                        csv_all = all_data.to_csv(index=False).encode('utf-8-sig')
                        
                        st.download_button(
                            label="전체 데이터",
                            data=csv_all,
                            file_name="전체_고객_예측결과.csv",
                            mime="text/csv",
                            help="전체 고객의 예측 결과를 다운로드합니다.",
                            use_container_width=True
                        )
                    
                    # 높은 위험 데이터 다운로드
                    with download_cols[1]:
                        high_risk_data = results_df[results_df['이탈 확률'] >= 0.7][display_columns].copy()
                        high_risk_data['이탈 확률'] = high_risk_data['이탈 확률'].apply(lambda x: f"{x:.1%}")
                        csv_high = high_risk_data.to_csv(index=False).encode('utf-8-sig')
                        
                        st.download_button(
                            label="높은 위험",
                            data=csv_high,
                            file_name="높은_위험_고객_예측결과.csv",
                            mime="text/csv",
                            help="이탈 확률이 70% 이상인 고객의 예측 결과를 다운로드합니다.",
                            use_container_width=True
                        )
                    
                    # 중간 위험 데이터 다운로드
                    with download_cols[2]:
                        medium_risk_data = results_df[
                            (results_df['이탈 확률'] >= 0.4) & 
                            (results_df['이탈 확률'] < 0.7)
                        ][display_columns].copy()
                        medium_risk_data['이탈 확률'] = medium_risk_data['이탈 확률'].apply(lambda x: f"{x:.1%}")
                        csv_medium = medium_risk_data.to_csv(index=False).encode('utf-8-sig')
                        
                        st.download_button(
                            label="중간 위험",
                            data=csv_medium,
                            file_name="중간_위험_고객_예측결과.csv",
                            mime="text/csv",
                            help="이탈 확률이 40-70%인 고객의 예측 결과를 다운로드합니다.",
                            use_container_width=True
                        )
                    
                    # 낮은 위험 데이터 다운로드
                    with download_cols[3]:
                        low_risk_data = results_df[results_df['이탈 확률'] < 0.4][display_columns].copy()
                        low_risk_data['이탈 확률'] = low_risk_data['이탈 확률'].apply(lambda x: f"{x:.1%}")
                        csv_low = low_risk_data.to_csv(index=False).encode('utf-8-sig')
                        
                        st.download_button(
                            label="낮은 위험",
                            data=csv_low,
                            file_name="낮은_위험_고객_예측결과.csv",
                            mime="text/csv",
                            help="이탈 확률이 40% 미만인 고객의 예측 결과를 다운로드합니다.",
                            use_container_width=True
                        )
            else:
                st.error('필터링된 데이터가 없습니다. 필터 조건을 조정해주세요.')

if __name__ == '__main__':
    main()

