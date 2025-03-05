from groq import Groq
from module.analysis_utils import generate_churn_analysis_data, generate_prompt_from_analysis
import os
# Groq API 키 설정
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# env에 키를 등록해야 함
# 방법: $env:GROQ_API_KEY= "your_api_key" (PowerShell)


# Groq 클라이언트 초기화
client = Groq(api_key=GROQ_API_KEY)

def get_churn_reasons_solutions(analysis_data):
    churn_analysis_prompt = generate_prompt_from_analysis(analysis_data)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": churn_analysis_prompt}],
        model="qwen-2.5-coder-32b",
    )
    return response.choices[0].message.content