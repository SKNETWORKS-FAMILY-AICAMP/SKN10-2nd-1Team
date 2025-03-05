import streamlit as st

def setup_filters(df):
    st.sidebar.header('필터 옵션')
    
    credit_score = st.sidebar.slider(
        '신용점수',
        int(df['credit_score'].min()),
        int(df['credit_score'].max()),
        (int(df['credit_score'].min()), int(df['credit_score'].max()))
    )
    
    age = st.sidebar.slider(
        '나이',
        int(df['age'].min()),
        int(df['age'].max()),
        (int(df['age'].min()), int(df['age'].max()))
    )
    
    tenure = st.sidebar.slider(
        '거래기간',
        int(df['tenure'].min()),
        int(df['tenure'].max()),
        (int(df['tenure'].min()), int(df['tenure'].max()))
    )
    
    balance = st.sidebar.slider(
        '계좌잔액',
        float(df['balance'].min()),
        float(df['balance'].max()),
        (float(df['balance'].min()), float(df['balance'].max()))
    )
    
    country = st.sidebar.multiselect(
        '국가',
        df['country'].unique().tolist(),
        default=df['country'].unique().tolist()
    )
    
    gender = st.sidebar.multiselect(
        '성별',
        df['gender'].unique().tolist(),
        default=df['gender'].unique().tolist()
    )
    
    products_number = st.sidebar.multiselect(
        '상품 수',
        df['products_number'].unique().tolist(),
        default=df['products_number'].unique().tolist()
    )
    
    credit_card = st.sidebar.multiselect(
        '신용카드 보유',
        [0, 1],
        default=[0, 1]
    )
    
    active_member = st.sidebar.multiselect(
        '활성 회원',
        [0, 1],
        default=[0, 1]
    )
    
    churn = st.sidebar.multiselect(
        '이탈 여부',
        [0, 1],
        default=[0, 1]
    )
    
    filters = {
        'credit_score': credit_score,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'country': country,
        'gender': gender,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'churn': churn
    }
    
    return filters

def filter_data(df, filters):
    filtered_df = df[
        (df['credit_score'].between(filters['credit_score'][0], filters['credit_score'][1])) &
        (df['age'].between(filters['age'][0], filters['age'][1])) &
        (df['tenure'].between(filters['tenure'][0], filters['tenure'][1])) &
        (df['balance'].between(filters['balance'][0], filters['balance'][1])) &
        (df['country'].isin(filters['country'])) &
        (df['gender'].isin(filters['gender'])) &
        (df['products_number'].isin(filters['products_number'])) &
        (df['credit_card'].isin(filters['credit_card'])) &
        (df['active_member'].isin(filters['active_member'])) &
        (df['churn'].isin(filters['churn']))
    ]
    
    return filtered_df