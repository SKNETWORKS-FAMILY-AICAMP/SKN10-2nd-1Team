import streamlit as st
import pandas as pd
import os

# 파일 경로 (VSCode에서 연결된 파일 기준)
file_path = "Bank Customer Churn Prediction(분석)2.xlsx"

def load_data():
    """ 엑셀 파일에서 Sheet1 데이터를 로드하고 정리하는 함수 """
    if not os.path.exists(file_path):
        st.error("❌ 파일을 찾을 수 없습니다. 파일을 업로드하세요.")
        return None
    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])  # 첫 번째 시트 기준
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

df = load_data()

if df is not None:
    # 중복된 컬럼명 정리 (의미 있는 이름 유지)
    def deduplicate_columns(columns):
        seen = {}
        new_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        return new_columns
    
    df.columns = deduplicate_columns(df.columns)
    
    # 불필요한 "Unnamed" 컬럼 제거 및 빈 값이 많은 행/열 삭제
    df_cleaned = df.dropna(axis=1, thresh=len(df) * 0.5)
    df_cleaned = df_cleaned.dropna(axis=0, thresh=len(df_cleaned.columns) * 0.5)
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains("Unnamed", na=False)]
    
    # ✅ 추가 수정: 개별 표 배치
    st.title("📊 Sheet1 분석")
    
    # 개별 표 배치
    st.markdown("### 📋 데이터 테이블")
    table_splits = [df_cleaned.iloc[i:i+10] for i in range(0, len(df_cleaned), 10)]
    for table in table_splits:
        st.table(table)
    
    # 분석 설명 추가
    st.markdown("""
    ### 🔍 **활동 고객과 이탈 관계**
    ✅ 활동 고객과 이탈은 **음의 상관관계** (group by 분석)
    🔴 비활동 고객이 금융 상품을 **이용하도록 혜택 제공 필요**
    
    ### 💳 **금융 상품과 이탈 관계**
    ✅ 신용카드 1개만 있는 고객이 **이탈 가능성 높음**
    🔴 금융 상품 2종 이상 유지 시 **이탈 감소** (입출금 통장 등 크로스셀링 필요)
    
    ### 👩‍💼 **성별과 이탈 분석**
    ✅ **여성 신용카드 고객**이 남성보다 이탈률 높음
    ✅ **프랑스 남성 고객**은 잔고가 높고 이탈률 가장 낮음
    
    ### 💰 **잔고 및 급여와 이탈**
    🔴 여성 고객이 남성보다 잔고 많지만 **이탈 더 많음** (고객 관리 필요)
    ✅ **프랑스 남성 고객**은 잔고 가장 높고 이탈률 가장 적음
    🔴 **독일 여성 고객**은 이탈 시 잔고가 높음 (고객 관리 필요)
    
    ### 🔄 **신용카드 활성 고객 분석**
    ✅ **장기 고객**의 이탈이 적음
    🔴 신용카드 **비활성 고객은 장기/단기 이탈률 높음** (상관관계 낮음)
    🔴 신용카드 활성 고객은 **단기 고객의 이탈이 높음** (장기 유지 필요)
    """)