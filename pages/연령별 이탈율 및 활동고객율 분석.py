import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 로드
def load_data():
    file_path = "./data/Bank Customer Churn Prediction(분석)2.xlsx"
    sheet_name = "Sheet2 (6)"
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=30)  # 30번째 행부터 데이터 시작
    df = df.iloc[:, [1, 2, 3]]  # 유효한 데이터 열 선택
    df.columns = ["Age", "Churn_Rate", "Active_Member"]
    df = df.dropna(subset=["Age"])  # 연령대 NaN 제거
    return df

# 데이터 처리
df = load_data()
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df = df.dropna(subset=["Age"])  # NaN 값 다시 제거
df.set_index("Age", inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# 스트림릿 앱
st.title("📊 연령별 이탈율 및 활동고객율 변화")

st.markdown(
    """
    ## 🔍 연령별 이탈율 및 활동고객율 분석
    - **18~39세:** 이탈율이 낮음.
    - **40~56세:** 이탈율이 계속 증가하여 고객 관리가 필요함.
    - **57~65세:** 이탈율이 점점 줄어들지만 여전히 높아 고객 관리가 필요함.
    
    ### 📌 활동 고객 비율 분석
    - **50세 이후:** 활동 고객 비율이 증가하는 경향을 보임.
    - **이전 나이대(50세 미만):** 활동 고객율이 약 50% 수준이므로 향상을 위한 전략 필요.
    """
)

# 그래프 생성
fig, ax = plt.subplots(figsize=(25, 7))  # 그래프 크기 조정

# X축 샘플링 조정
sample_rate = max(1, len(df) // 70)  # 70개 이하의 점만 표시
sampled_df = df.iloc[::sample_rate]

# X축 레이블(연령대)
x = np.arange(len(sampled_df.index))  # 연령 인덱스 생성
width = 0.35  # 막대 너비 조정

# 여러 개의 데이터 세트 플로팅
df_columns = ["Churn_Rate", "Active_Member"]
colors = ['#ED7D31', '#5B9BD5']  # 원본 엑셀 색상 적용 (주황: 이탈율, 파랑: 활동 고객율)
labels = ["이탈율 (Churn Rate)", "활동 고객율 (Active Member)"]

for i, col in enumerate(df_columns):
    ax.bar(x + i * width, sampled_df[col], width=width, label=labels[i], color=colors[i])

# 그래프 설정
ax.set_title("연령별 이탈율 및 활동고객율 변화", fontsize=16, fontweight='bold')
ax.set_xlabel("연령", fontsize=12)
ax.set_ylabel("비율", fontsize=12)
ax.set_xticks(x + width * 0.5)  # x축 정렬
ax.set_xticklabels(sampled_df.index.astype(int), rotation=45, fontsize=10)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

st.pyplot(fig)