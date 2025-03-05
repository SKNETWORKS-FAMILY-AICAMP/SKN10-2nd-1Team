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
    돈단위는 유로(€)로 표기합니다.
    참고해야되는 정보는 다음과같아.
    🔹 기본 정보
    - 총 고객 수: {sum(analysis_data["risk_counts"].values())}명
    - 높은 위험: 이탈 확률 70% 이상
    - 중간 위험: 이탈 확률 40% ~ 70% 미만
    - 낮은 위험: 이탈 확률 40% 미만
    - 높은 위험 고객 수: {analysis_data["risk_counts"].get("높음", 0)}명
    - 중간 위험 고객 수: {analysis_data["risk_counts"].get("중간", 0)}명
    - 낮은 위험 고객 수: {analysis_data["risk_counts"].get("낮음", 0)}명

    🔹 위험 수준별 고객 특성
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

    🔹 연령대별, 국가별, 성별 이탈률
    📌 **연령대별 이탈률 (%)**
    {analysis_data["age_churn_rates"]}

    📌 **국가별 이탈률 (%)**
    {analysis_data["country_churn_rates"]}

    📌 **성별 이탈률 (%)**
    {analysis_data["gender_churn_rates"]}


    
    ### 출력 형식 (항상 이 형식 유지)
    원인이 수치적 데이터와 관련있다면 수치적으로 분석할것
    출력은 아래 예시와 같은 형식으로만 제공해야함 (형식 외 다른 문장 출력 금지)

    
    - **연령대** (원인1)
        - 설명: 연령대에 따른 이탈률을 보면, 특히 40대와 50대 고객의 이탈률이 높은 것으로 나타났습니다. 특히 50대 고객의 이탈률이 54.44%로 가장 높습니다. 이는 이 연령대의 고객들이 퇴직 기간에 다가 다가오거나, 또는 건강 관리 위한 비용 증가 등으로 인해 우리 은행에서 자금을 이탈하거나 이용을 중단할 가능성이 높다는 것을 시사할 수 있습니다.
        - 해결방안: 이러한 연령대의 고객들에게는 세금 및 재산 관리, 건강 관련 보험 서비스가 필요한 것으로 추정됩니다. 따라서 은행이 이러한 제안을 제공하거나 적극적인 고객 관계 관리로 신용도를 높이고, 리테일 은행 서비스 외에 생명보험, 건강보험 제품 등을 개발하거나 제공하여 고객의 긍정적인 이직률을 유지해야 합니다.
    - **국가** (원인2)
        - 설명: 국가는 기대 이탈률과 관련이 있습니다. 독일의 경우 특히 높은 이탈률(32.73%)을 보였는데 이는 독일 문화 중 금융 서비스의 다양성과 선택의 폭이 좁은 경우 더 높은 이탈률을 보이는 경향이 있으며, 이는 유럽의 경쟁 환경에서 독일이 비교적 낮은 점유율을 보이는 한 가지 그 원인이 될 수 있습니다.
        - 해결방안: 독일을 포함한 대상 국가들에 대한 조사를 통해 시장 특성을 파악하고 더욱 맞춤형 금융 서비스를 제공해야 합니다. 추가적으로 거대한 유럽 시장에서 경쟁 우위를 가져올 수 있는 혁신적인 제품을 개발하거나, 파트너십을 맺기 위한 다양한 네트워크 구축에도 노력해야 합니다.
    - **성별** (원인3)
        - 설명: 성별로 보면 여성이 남성보다 이탈률이 높습니다 (여성 24.50%, 남성 17.67%). 이는 여성이 일반적으로 남성보다 금융 상담을 받는 편이고, 이 과정에서 혼란스러움을 경험할 가능성이 높아서 그렇습니다. 또한 여성의 금융 정보 접근성이나 금융 상담 만족도가 낮을 가능성이 큽니다.
        - 해결방안: 그려본 문제를 해결하기 위해서는 성별에 따른 금융 교육 프로그램을 강화하거나 다양한 성별 고객을 대상으로 금융 상담 서비스를 제공해야 합니다. 또한 여성이 더 많이 활용하는 디지털 채널을 통해 금융 서비스를 제공하여 여성이 금융 상담에 대한 접근성을 높이고, 성별 차별을 없애는 문화적 변화를 적극적으로 유도해야 합니다.
    """




    return prompt