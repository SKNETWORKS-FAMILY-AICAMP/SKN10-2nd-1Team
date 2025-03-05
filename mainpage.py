import streamlit as st
import pandas as pd
import joblib
import warnings
import pickle
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from module.inho_model import load_data
from module.churn_prediction import predict_churn
from module.analysis_utils import generate_churn_analysis_data, generate_prompt_from_analysis
from module.groq_utils import get_churn_reasons_solutions
from module.filter_utils import setup_filters, filter_data  # Import the new module
from module.display_utils import display_metrics, display_risk_customers, calculate_risk_info  # Import the new module
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 사용자
plt.rcParams['font.size'] = 12

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

def main():
    st.title('은행 고객 이탈 예측 시스템')
    
    # 데이터 로드
    df = load_data('./data/Bank Customer Churn Prediction.csv')
    
    # 표시할 컬럼 설정
    display_columns = ['customer_id', 'country', 'age', 'balance', '이탈 예측', '이탈 확률']
    
    # 세션 상태 초기화
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    # 필터 설정
    filters = setup_filters(df)
    
    # 데이터 필터링
    filtered_df = filter_data(df, filters)
    
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
                    
                    # 메트릭 표시
                    display_metrics(total_customers, predicted_churns, churn_rate, high_risk, medium_risk, low_risk)

                    # 구분선 추가
                    st.markdown("---")
                    
                    # 위험도별 고객 목록 표시
                    display_risk_customers(results_df, display_columns)
                    
                    # 각 위험 수준에 속한 고객들의 정보 계산
                    high_risk_info = calculate_risk_info(results_df[results_df['이탈 확률'] >= 0.7])
                    medium_risk_info = calculate_risk_info(results_df[(results_df['이탈 확률'] >= 0.4) & (results_df['이탈 확률'] < 0.7)])
                    low_risk_info = calculate_risk_info(results_df[results_df['이탈 확률'] < 0.4])
                    
                    # 분석 데이터 생성
                    analysis_data = generate_churn_analysis_data(results_df)

                    # Groq API 요청
                    churn_reasons_solutions = get_churn_reasons_solutions(analysis_data)

                    # Streamlit에 표시
                    st.markdown("### 고객 이탈 원인 및 해결 방안")
                    st.markdown(churn_reasons_solutions)

                    # 위험도 기준 설명
                    st.markdown("""
                    ### 위험도 기준
                    - 🔴 높은 위험: 이탈 확률 70% 이상
                    - 🟡 중간 위험: 이탈 확률 40% ~ 70% 미만
                    - 🟢 낮은 위험: 이탈 확률 40% 미만
                    """)

                    # 구분선 추가
                    st.markdown("---")
                    
            else:
                st.error('필터링된 데이터가 없습니다. 필터 조건을 조정해주세요.')

if __name__ == '__main__':
    main()