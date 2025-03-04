import pandas as pd

def generate_churn_analysis_data(results_df):
    """ 모델 예측 결과 기반으로 위험도별 주요 분석 데이터를 생성 """
    
    results_df["risk_level"] = pd.cut(results_df["이탈 확률"], bins=[0, 0.4, 0.7, 1.0], labels=["낮음", "중간", "높음"])
    risk_counts = results_df["risk_level"].value_counts().to_dict()
    risk_group_means = results_df.groupby("risk_level")[["credit_score", "balance", "estimated_salary"]].mean().to_dict()

    results_df["age_group"] = pd.cut(results_df["age"], bins=[18, 30, 40, 50, 60, 100], labels=["20대", "30대", "40대", "50대", "60대 이상"])
    age_churn_rates = results_df.groupby("age_group")["이탈 확률"].mean() * 100
    country_churn_rates = results_df.groupby("country")["이탈 확률"].mean() * 100
    gender_churn_rates = results_df.groupby("gender")["이탈 확률"].mean() * 100

    return {
        "risk_counts": risk_counts,
        "risk_group_means": risk_group_means,
        "age_churn_rates": age_churn_rates.to_dict(),
        "country_churn_rates": country_churn_rates.to_dict(),
        "gender_churn_rates": gender_churn_rates.to_dict()
    }

def generate_prompt_from_analysis(analysis_data):
    """ 분석된 데이터를 바탕으로 Groq API 요청을 위한 프롬프트 생성 """
    
    prompt = f"""
    ### 고객 이탈 분석 요청 (한국어로 작성)
    주어진 데이터를 바탕으로 고객 이탈 원인과 해결 방안을 도출하시오.
    # 돈단위는 유로(€)로 표기합니다.
    ### 🔹 기본 정보
    - 총 고객 수: {sum(analysis_data["risk_counts"].values())}명
    - 높은 위험: 이탈 확률 70% 이상
    - 중간 위험: 이탈 확률 40% ~ 70% 미만
    - 낮은 위험: 이탈 확률 40% 미만
    - 높은 위험 고객 수: {analysis_data["risk_counts"].get("높음", 0)}명
    - 중간 위험 고객 수: {analysis_data["risk_counts"].get("중간", 0)}명
    - 낮은 위험 고객 수: {analysis_data["risk_counts"].get("낮음", 0)}명

    ### 🔹 위험 수준별 고객 특성
    📌 **높은 위험 고객**
    - 평균 신용 점수: {analysis_data["risk_group_means"]["credit_score"].get("높음", "N/A")}
    - 평균 계좌 잔액: {analysis_data["risk_group_means"]["balance"].get("높음", "N/A")}
    - 평균 연봉: {analysis_data["risk_group_means"]["estimated_salary"].get("높음", "N/A")}

    📌 **중간 위험 고객**
    - 평균 신용 점수: {analysis_data["risk_group_means"]["credit_score"].get("중간", "N/A")}
    - 평균 계좌 잔액: {analysis_data["risk_group_means"]["balance"].get("중간", "N/A")}
    - 평균 연봉: {analysis_data["risk_group_means"]["estimated_salary"].get("중간", "N/A")}

    📌 **낮은 위험 고객**
    - 평균 신용 점수: {analysis_data["risk_group_means"]["credit_score"].get("낮음", "N/A")}
    - 평균 계좌 잔액: {analysis_data["risk_group_means"]["balance"].get("낮음", "N/A")}
    - 평균 연봉: {analysis_data["risk_group_means"]["estimated_salary"].get("낮음", "N/A")}

    ### 🔹 연령대별, 국가별, 성별 이탈률
    📌 **연령대별 이탈률 (%)**
    {analysis_data["age_churn_rates"]}

    📌 **국가별 이탈률 (%)**
    {analysis_data["country_churn_rates"]}

    📌 **성별 이탈률 (%)**
    {analysis_data["gender_churn_rates"]}


    ### 응답 형식 (항상 이 형식 유지)
    원인이 수치적 데이터와 관련있다면 수치적으로 분석할것

    - 원인 1
        - 설명
        - 해결방안
    - 원인 2
        - 설명
        - 해결방안
    - 원인 3
        - 설명
        - 해결방안

    """


    return prompt