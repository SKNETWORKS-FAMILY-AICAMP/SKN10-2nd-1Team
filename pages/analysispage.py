import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 사용자
plt.rcParams['font.size'] = 12

# 파일 로드
def load_data():
    file_path = "Bank Customer Churn Prediction(분석)2.xlsx"
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name='Bank Customer Churn Prediction')
    return df

df = load_data()

# 페이지 제목
st.title("은행 고객 이탈 분석")

# 데이터 개요 보기
if st.checkbox("데이터 미리보기"):
    st.write(df.head())

# 데이터 통계 요약
if st.checkbox("통계 요약 보기"):
    st.write(df.describe())

# 고객 이탈 분포 시각화
st.subheader("고객 이탈 현황")
churn_counts = df['churn'].value_counts()
fig, ax = plt.subplots()
labels = ["유지", "이탈"]
colors = ['lightblue', 'salmon']
explode = (0, 0.1)  # 이탈 부분 강조
ax.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
plt.axis('equal')  # 원형 유지
st.pyplot(fig)

# 이탈 고객 분석
st.subheader("특성별 이탈 분석")
feature = st.selectbox("분석할 특성을 선택하세요", ["credit_score", "age", "balance", "products_number", "estimated_salary"])
fig, ax = plt.subplots()
sns.boxplot(x='churn', y=feature, data=df, ax=ax)
st.pyplot(fig)

# 특정 고객 필터링
gender = st.selectbox("성별 선택", df['gender'].unique())
country = st.selectbox("국가 선택", df['country'].unique())
filtered_df = df[(df['gender'] == gender) & (df['country'] == country)]
st.write(filtered_df.head())
