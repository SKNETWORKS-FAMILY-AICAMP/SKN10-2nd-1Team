import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 로드
def load_data():
    file_path = "Bank Customer Churn Prediction(분석)2.xlsx"
    sheet_name = "Sheet2 (4)"
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=31)  # 31번째 행부터 데이터 시작
    df = df.iloc[:, [1, 2, 3, 4, 6, 7]]  # 유효한 데이터 열 선택
    df.columns = ["Age", "Balance", "Salary", "Prev_Balance", "Total_Balance", "Total_Salary"]
    df = df.dropna(subset=["Age"])  # 연령대 NaN 제거
    return df

# 데이터 처리
df = load_data()
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df = df.dropna(subset=["Age"])  # NaN 값 다시 제거
df.set_index("Age", inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# 스트림릿 앱
st.title("📊 Sheet2 (4) 연령별 잔고 및 연봉 변화")

st.markdown(
    """
    ## 🔍 연령별 연봉 및 잔고 변화 분석
    - **유지 고객 중 34~38세**의 연봉 및 잔고가 가장 높으며, 이후 점점 감소합니다.
    - **이탈 고객 중 40대**의 연봉 및 잔고가 가장 높으며, 이후 점점 감소하는 경향을 보입니다.
    
    ### 📌 연령대별 차이 분석
    - **18~48세:** 유지 고객이 이탈 고객보다 연봉 및 잔고가 더 많음.
    - **49~60세:** 유지 고객이 이탈 고객보다 연봉 및 잔고가 더 적음 (**고객 관리 필요**).
    - **61~92세:** 유지 고객이 이탈 고객보다 연봉 및 잔고가 더 많음.
    """
)

# 그래프 생성
fig, ax = plt.subplots(figsize=(25, 7))  # 그래프 크기 조정

# X축 샘플링 조정 (더 넓게 표시)
sample_rate = max(1, len(df) // 60)  # 60개 이하의 점만 표시
sampled_df = df.iloc[::sample_rate]

# X축 레이블(연령대)
x = np.arange(len(sampled_df.index))  # 연령 인덱스 생성
width = 0.2  # 막대 너비 조정

# 여러 개의 데이터 세트 플로팅
df_columns = ["Balance", "Salary", "Prev_Balance", "Total_Balance", "Total_Salary"]
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#70AD47']  # 엑셀 원본 색상 적용
labels = ["잔고(Balance)", "연봉(Salary)", "과거 잔고(Prev_Balance)", "총 잔고(Total_Balance)", "총 연봉(Total_Salary)"]

for i, col in enumerate(df_columns):
    ax.bar(x + i * width, sampled_df[col], width=width, label=labels[i], color=colors[i])

# 그래프 설정
ax.set_title("연령별 잔고 및 연봉 변화", fontsize=16, fontweight='bold')
ax.set_xlabel("연령", fontsize=12)
ax.set_ylabel("금액", fontsize=12)
ax.set_xticks(x + width * 2)  # x축 정렬
ax.set_xticklabels(sampled_df.index.astype(int), rotation=45, fontsize=10)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

st.pyplot(fig)