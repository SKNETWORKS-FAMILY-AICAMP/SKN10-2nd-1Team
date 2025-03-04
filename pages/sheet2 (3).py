import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 파일 로드
def load_data():
    file_path = "Bank Customer Churn Prediction(분석)2.xlsx"
    sheet_name = "Sheet2 (3)"
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=30)  # 31번째 행부터 데이터 시작
    df = df.iloc[:, [1, 3, 4]]  # 유효한 데이터 열 선택
    df.columns = ["Credit Score", "Churn", "Total"]
    df = df.dropna(subset=["Credit Score"])  # 신용점수 NaN 제거
    return df

# 데이터 처리
df = load_data()
df["Credit Score"] = pd.to_numeric(df["Credit Score"], errors='coerce')
df = df.dropna(subset=["Credit Score"])  # NaN 값 다시 제거

df.set_index("Credit Score", inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# 스트림릿 앱
st.title("📊 Sheet2 (3) 신용점수와 이탈수 관계")

st.markdown(
    """
    ## 🔍 데이터 분석 요약
    - **신용점수 350점(최저점)**도 신용카드를 보유한 고객이 존재합니다.
    - **신용점수 350~404점** 사이의 고객들은 **전부 이탈**하는 경향을 보입니다.
    - **그 외 신용점수와 이탈수의 상관관계는 크지 않은 것으로 분석됩니다.**
    - **신용점수 850점(만점) 고객**이 과도하게 포진되어 있어, **이상치로 고려할 가능성이 있습니다.**
    """
)

fig, ax = plt.subplots(figsize=(15, 7))  # 그래프 크기 확대
df.plot(kind="bar", ax=ax, width=0.8)
ax.set_title("신용점수와 이탈수 관계", fontsize=14, fontweight='bold')
ax.set_xlabel("신용점수 (일부만 표시)", fontsize=12)
ax.set_ylabel("이탈 수", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.set_xticks(range(0, len(df), max(1, len(df) // 20)))  # 너무 촘촘한 경우 간격 조정
ax.set_xticklabels(df.index.dropna().astype(int)[::max(1, len(df) // 20)], rotation=45, fontsize=10)  # NaN 제거 후 정수 변환

st.pyplot(fig)