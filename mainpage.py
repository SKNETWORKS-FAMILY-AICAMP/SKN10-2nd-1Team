import streamlit as st
import pandas as pd
import joblib
import warnings
import pickle
from sklearn.preprocessing import PowerTransformer, StandardScaler
warnings.filterwarnings("ignore")

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


def predict_churn(filtered_data, model_select:str, df=pd.read_csv('./data/Bank Customer Churn Prediction.csv')):  # df 하드코딩 했습니다
    if model_select == '민경':
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
    
    elif model_select == '윤홍':
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

def main():
    st.title('은행 고객 이탈 예측 시스템')
    
    # 데이터 로드
    df = pd.read_csv('./data/Bank Customer Churn Prediction.csv')
    
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
        (df['active_member'].isin(active_member))
    ]
    
    # 필터링된 데이터 표시
    st.write(f"필터링된 고객 수: {len(filtered_df):,}명")
    st.dataframe(filtered_df)
    
    # 예측 버튼과 모델 선택 박스
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        model_select = st.selectbox('모델 선택', ['민경', '윤홍'], index=0)
        if st.button('이탈 예측하기', use_container_width=True):
            if len(filtered_df) > 0:
                with st.spinner('예측 중...'):
                    predictions, probabilities = predict_churn(filtered_df, model_select)
                    
                    # 결과를 데이터프레임에 추가
                    results_df = filtered_df.copy()
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
                    
                    # 위험도 기준 설명
                    st.markdown("""
                    ### 위험도 기준
                    - 🔴 높은 위험: 이탈 확률 70% 이상
                    - 🟡 중간 위험: 이탈 확률 40% ~ 70% 미만
                    - 🟢 낮은 위험: 이탈 확률 40% 미만
                    """)
                    
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
