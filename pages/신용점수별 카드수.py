
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 로드 함수
def load_data():
    file_path = "Bank Customer Churn Prediction(분석)2.xlsx"
    sheet_name = "Bank Customer Churn Prediction"
    
    # 데이터 읽기
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=30)
    
    # 필요한 열 선택 (열 이름으로 선택)
    try:
        df = df.iloc[:, [1, 4, 8, 11]]  # 유효한 데이터 열 선택
        df.columns = ["credit_score", "age", "credit_card", "churn"]
    except KeyError as e:
        st.error(f"열 이름이 유효하지 않습니다: {e}")
        st.stop()
    
    # 결측값 제거
    df = df.dropna(subset=["credit_score"])  # 신용점수 NaN 제거
    
    return df

# 데이터 처리
df = load_data()

# churn 값에 따라 데이터 분리
df_churn_1 = df[df['churn'] == 1]
df_churn_0 = df[df['churn'] == 0]

# 막대그래프 생성 함수 (두 데이터를 하나의 그래프에 표시)
def plot_combined_bar(df_churn_0, df_churn_1):
    # churn=0 데이터 그룹화
    grouped_data_0 = df_churn_0.groupby('credit_score')['credit_card'].sum().reset_index()
    grouped_data_1 = df_churn_1.groupby('credit_score')['credit_card'].sum().reset_index()

    # 두 그룹의 신용점수를 동일한 x축에 맞추기 위해 병합
    combined_data = pd.merge(grouped_data_0, grouped_data_1, on='credit_score', how='outer', suffixes=('_churn_0', '_churn_1')).fillna(0)

    credit_scores = combined_data['credit_score']
    credit_cards_0 = combined_data['credit_card_churn_0']
    credit_cards_1 = combined_data['credit_card_churn_1']

    x = np.arange(len(credit_scores))  # x축 위치 설정
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 막대그래프 생성 (churn=0과 churn=1 각각)
    ax.bar(x - bar_width/2, credit_cards_0, bar_width, label='churn=0', color='orange')
    ax.bar(x + bar_width/2, credit_cards_1, bar_width, label='churn=1', color='skyblue')

    # 그래프 설정
    ax.set_xlabel('Credit Score')
    ax.set_ylabel('Total Number of Credit Cards')
    ax.set_title('Credit Score vs Credit Cards (churn=0 and churn=1)')

    # x축 레이블 간격 조정 (최대 20개만 표시)
    step_size = max(1, len(credit_scores) // 20)
    ax.set_xticks(x[::step_size])
    ax.set_xticklabels(credit_scores[::step_size].astype(int), rotation=45)  # 신용점수를 정수로 변환하여 표시
    
    ax.legend()
    return fig

# 스트림릿 앱
st.title("📊 신용점수와 이탈수 관계")

st.markdown(
    """
    ## 🔍 데이터 분석 요약
    - **신용점수 350점(최저점)** 도 신용카드를 보유한 고객이 존재합니다.
    - **신용점수 350~404점** 사이의 고객들은 **전부 이탈**하는 경향을 보입니다.
    - **그 외 신용점수와 이탈수의 상관관계는 크지 않은 것으로 분석됩니다.**
    - **신용점수 850점(만점) 고객** 이 과도하게 포진되어 있어, **이상치로 고려해야할 수 있습니다.**
    """
)
# 그래프 생성 및 표시
fig = plot_combined_bar(df_churn_0, df_churn_1)
st.pyplot(fig)