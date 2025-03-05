import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 로드 함수
def load_data():
    file_path = "./data/Bank Customer Churn Prediction(분석)2.xlsx"
    sheet_name = "Bank Customer Churn Prediction"
    
    # 데이터 읽기
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=30)
    
    # 데이터프레임 구조 확인 (Streamlit 앱 실행 중에는 출력되지 않으므로 주석 처리 가능)
    # st.write(df.head())
    
    # 필요한 열 선택 (열 이름으로 선택)
    try:
        #df = df[["credit_score", "age", "credit_card", "churn"]]
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

# 막대그래프 생성 함수
def plot_grouped_bar(df_subset, churn_value, color):
    grouped_data = df_subset.groupby('age')['credit_score'].mean().reset_index()

    ages = grouped_data['age']
    credit_scores = grouped_data['credit_score']

    x = np.arange(len(ages))  # x축 위치 설정
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(36, 12))
    ax.bar(x, credit_scores, bar_width, label='Credit Score', color=color)

    # 그래프 설정
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Credit Score')
    ax.set_title(f'Grouped Bar Chart for churn = {churn_value}')
    ax.set_xticks(x)
    ax.set_xticklabels(ages.astype(int), rotation=45)  # 나이를 정수로 변환하여 표시
    ax.legend()

    return fig

# 스트림릿 앱 시작
st.title("📊 나이별 평균신용점수")

st.markdown(
    """
    ## 🔍 데이터 분석 요약
    - **나이, 신용점수, 이탈 사이에는 상관관계가 부족함**
    """
)

# 그래프 생성 및 표시
fig1 = plot_grouped_bar(df_churn_1, churn_value=1, color='skyblue')
fig2 = plot_grouped_bar(df_churn_0, churn_value=0, color='orange')

st.pyplot(fig1)
st.pyplot(fig2)