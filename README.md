# SKN10-2nd-1Team
# [가입 고객 이탈 예측](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
 SK Networks AI Camp 10기

 개발기간: 25.02.19 - 25.03.05
<br>

# 0. 팀 소개

### 팀명 : 1 팀
### 팀원 소개
<table align=center>
<tbody>
 <tr>
  <br>
      <td align=center><b>배민경👑</b></td>
      <td align=center><b>장윤홍</b></td>
      <td align=center><b>이유호</b></td>
      <td align=center><b>남궁세정</b></td>
      <td align=center><b>황인호</b></td>
    </tr>
    <br>
  <tr>
      <td><a href="https://github.com/baeminkyeong"><div align=center>@baeminkyeong</div></a></td>
      <td><a href="https://github.com/yuuunong"><div align=center>@yuuunong</div></a></td>
      <td><a href="https://github.com/netsma"><div align=center>@netsma</div></a></td>
      <td><a href="https://github.com/petoriko"><div align=center>@petoriko</div></a></td>
      <td><a href="https://github.com/HIHO999"><div align=center>@HIHO999</div></a></td>
    </tr>
     </tr>
   </tbody>
</table>
<br>


# 1. 프로젝트 개요

### 프로젝트
- 은행 가입고객 이탈자 분석 및 예측

### 목표
- 본 프로젝트는 데이터 분석 및 머신러닝, 딥러닝을 활용하여 **은행 고객의 이탈 가능성을 예측하는 모델**을 개발하는 것입니다.

### 프로젝트 배경

![alt text](<img/스크린샷 2025-03-04 152946.png>)

하나금융연구소 - "2024년, 은행이 놓치지 말아야 할 3가지" 장혜원 수석연구원

- 금융 시장에서 고객관계 강화는 은행의 최우선 과제 중 하나입니다.

- 그러나 디지털 전환 비용과 함께 다양한 경쟁자 참여로 전통적인 금융기관의 마케팅 비용은 매해 증가하는 반면, 고객 충성도는 하락하고 있는 상황



<br>

![alt text](img/image.png)

매경이코노미 -"[경영칼럼] 신규 고객 늘리기보다 기존 고객 유지 힘써라" 이성용

- **기존 고객 유지를 하는 것이 신규 고객을 유치하는 것보다 수익성 5 ~ 7배 향상**된다고 알려져 있습니다.

- 따라서, 기존 고객의 이탈을 방지하는 것이 운영 비용 절감 및 수익성 강화에 효과적인 전략이 될 수 있습니다.

- 이에 따라, 사전적으로 고객 이탈을 예측하고 선제적으로 대응할 수 있는 데이터 기반의 고객 이탈 예측 모델이 필요하게 되었습니다.

<br>

### 기대 효과
| 기대효과 |내용|
|------|---|
|고객이탈|이탈 가능성이 높은 고객을 조기에 발견하여 맞춤형 프로모션 및 상담 제공|
|비용절감|고객 유지 비용 절감 및 신규 고객 유치 비용 최소화|
|비즈니스 성장|데이터 기반 의사결정을 통한 은행의 경쟁력 강화 및 고객 만족도 향상|

### 요약
- 본 프로젝트를 통해 은행은 고객 이탈 문제를 보다 효과적으로 해결하고, **장기적인 고객 관계 관리를 강화**할 수 있습니다.

- 데이터 기반의 **예측 모델을 활용**하여 고객 맞춤형 전략을 수립함으로써 전통적인 은행의 지속 가능한 성장을 도모하는 것이 본 프로젝트의 최종 목표입니다.


# 2. 기술 스택

| 분야 |기술|
|------|---|
|협업 및 형상 관리|<img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=Discord&logoColor=white" /> <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white" /> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white" />|
|개발 환경 & 언어|<img src="https://img.shields.io/badge/VScode-007ACC?style=for-the-badge&logo=Visual-Studio-Code&logoColor=white" /> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white" />|
|데이터 분석 & 학습|<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white" /> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=NumPy&logoColor=white" /> <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=Matplotlib&logoColor=white" /> <img src="https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge&logo=Seaborn&logoColor=white" /> <img src="https://img.shields.io/badge/Scikit%20Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />|
|대시보드|<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />|

# 3. 데이터 전처리 
- ABC 은행의 고객 이탈 데이터 <br>
- 출처: https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset
<br>

 **1) 데이터 내용 확인**

| 변수명             | 변수 설명                                             | 변수 타입   |
|-------------------|----------------------------------------------------|------------------|
| customer_id       | 고객을 구별하는 고유 ID               | object      |
| credit_score      | 고객의 신용 점수                    | int64            |
| country           | 고객이 거주하는 국가                | object (범주형)   |
| gender            | 고객의 성별                        | object (범주형)   |
| age               | 고객의 나이                        | int64            |
| tenure            | 고객의 은행 가입 기간             | int64            |
| balance           | 고객의 은행 잔액                  | float64          |
| products_number   | 고객이 보유한 은행 상품 수        | int64            |
| credit_card       | 고객의 신용카드 보유 여부    | int64 (범주형)     |
| active_member     | 고객의 활성 회원 여부       | int64 (범주형)     |
| estimated_salary  | 고객의 추정 급여                   | float64          |
| churn             | 고객의 이탈 여부  | int64 (범주형)     |

- 변수 : credit_score (신용 점수), country (국가), age (나이), tenure (가입 기간), churn (이탈 여부) 등의 변수 <br>
- 데이터 크기: 총 10,000명의 고객 데이터, 12개의 변수 (2개의 object형 변수, 8개의 int형 변수, 2개의 float형 변수) <br>
- 데이터 유형: 5개의 범주형 데이터, 7개의 수치형 데이터
  
 **2) 결측치 확인**
 
 ![결측치](./img/Missing_values.png)
 
- 결측치 확인 결과 : 결측치가 없음
  
 **3) 데이터 분석**
 
![이탈률](./img/churn.png) 
![나라](./img/country.png)
![성별](./img/gender.png)
![age](./img/age.png)
![credit_score](./img/credit_score.png)
![credit_card](./img/credit_card.png)
![products_num](./img/products_num.png)
![tenure](./img/tenure.png)
![balance](./img/balance.png)
![hitmap](./img/hitmap.png)
![full_count](./img/full_count.png)
![GB_feature](./img/GB_feature.png)
![RF_feature](./img/RF_feature.png)



# 4. 실행 결과

# 5.  한줄 회고
