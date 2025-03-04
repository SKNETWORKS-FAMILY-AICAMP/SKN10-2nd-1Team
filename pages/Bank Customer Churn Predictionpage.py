import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 사용자
plt.rcParams['font.size'] = 12

# 파일 로드 함수
def load_data():
    file_path = "Bank Customer Churn Prediction(분석)2.xlsx"
    if not os.path.exists(file_path):
        st.error("파일을 찾을 수 없습니다. 파일을 업로드하세요.")
        return None
    
    try:
        xls = pd.ExcelFile(file_path)
        if "Bank Customer Churn Prediction" in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name="Bank Customer Churn Prediction")
            return df
        else:
            st.error("❌ 'Bank Customer Churn Prediction' 시트를 찾을 수 없습니다.")
            return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

df = load_data()

if df is not None:
    # 데이터 정리: 첫 번째 행이 컬럼명일 경우 수정
    if df.iloc[0].isna().sum() == 0:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    st.title("📊 Bank Customer Churn Prediction 분석")

    # 데이터 개요 보기
    if st.checkbox("🔍 데이터 미리보기"):
        st.dataframe(df)

    # 1. 고객 이탈 현황 분석
    if 'churn' in df.columns:
        st.subheader("📌 고객 이탈 현황")
        churn_counts = df['churn'].value_counts()

        if len(churn_counts) > 1:
            labels = churn_counts.index.astype(str)
            fig, ax = plt.subplots()
            colors = ['lightblue', 'salmon']
            explode = (0, 0.1) if len(labels) == 2 else None
            ax.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], explode=explode, startangle=90)
            plt.axis('equal')
            st.pyplot(fig)
        else:
            st.write("⚠️ 데이터가 하나의 클래스만 포함하고 있어 바 차트를 표시합니다.")
            fig, ax = plt.subplots()
            churn_counts.plot(kind='bar', color='salmon', ax=ax)
            ax.set_ylabel("고객 수")
            st.pyplot(fig)

        st.write("📌 **여성 고객이 남성 고객보다 급여가 더 많음에도 이탈을 더 많이 함. (고객 관리 필요)**")

    # 2. 성별별 고객 이탈률
    if 'gender' in df.columns and 'churn' in df.columns:
        st.subheader("📌 성별별 고객 이탈률")
        gender_churn = df.groupby("gender")["churn"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        gender_churn.plot(kind='bar', color=["blue", "red"], ax=ax)
        ax.set_ylabel("이탈률")
        st.pyplot(fig)

    # 3. 연령대별 잔고 및 연봉 분석 (유지 고객만)
    if 'age' in df.columns and 'balance' in df.columns and 'estimated_salary' in df.columns:
        st.subheader("📌 연령대별 잔고 및 연봉 분석 (유지 고객)")
        age_bins = list(range(18, 80, 5))
        df["age_group"] = pd.cut(df["age"], bins=age_bins, right=False)

        if 'churn' in df.columns:
            churn_0 = df[df["churn"] == 0]
            balance_avg = churn_0.groupby("age_group")["balance"].mean()
            salary_avg = churn_0.groupby("age_group")["estimated_salary"].mean()

            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(balance_avg.index.astype(str), balance_avg, marker="o", linestyle="-", color="blue", label="잔고")
            ax1.set_ylabel("잔고", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.plot(salary_avg.index.astype(str), salary_avg, marker="o", linestyle="--", color="red", label="연봉")
            ax2.set_ylabel("연봉", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            plt.title("연령대별 잔고 및 연봉 변화 (유지 고객)")
            st.pyplot(fig)

            st.write("📌 **34~38세 유지 고객의 잔고 및 연봉이 가장 높고 이후 점차 감소함.**")

    # 4. 고객 이탈 분석 - 신용 점수
    if 'churn' in df.columns and 'credit_score' in df.columns:
        st.subheader("📌 이탈 여부에 따른 신용 점수 분포")
        fig, ax = plt.subplots()
        sns.boxplot(x='churn', y='credit_score', data=df, ax=ax)
        st.pyplot(fig)

        st.write("📌 **이탈 고객의 평균 신용 점수가 유지 고객보다 낮음.**")

    # 5. 국가별 고객 이탈률
    if 'country' in df.columns and 'churn' in df.columns:
        st.subheader("📌 국가별 고객 이탈률")
        country_churn = df.groupby("country")["churn"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        country_churn.plot(kind='bar', color="salmon", ax=ax)
        ax.set_ylabel("이탈률")
        st.pyplot(fig)

    # 6. 연령대별 고객 분포 (추가된 그래프)
    if 'age' in df.columns:
        st.subheader("📌 연령대별 고객 분포")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["age"], bins=20, kde=True, color="purple", ax=ax)
        ax.set_xlabel("나이")
        ax.set_ylabel("고객 수")
        st.pyplot(fig)

        st.write("📌 **연령대별로 보면 특정 연령층에서 고객 수가 집중됨.**")
