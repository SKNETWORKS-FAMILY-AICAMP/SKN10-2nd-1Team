import streamlit as st

def display_metrics(total_customers, predicted_churns, churn_rate, high_risk, medium_risk, low_risk):
    st.markdown("### 예측 결과 요약")
    
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    [data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("전체 고객", f"{total_customers:,}명")
    
    with col2:
        st.metric("이탈 예정", f"{predicted_churns:,}명", f"{churn_rate:.1f}%")
    
    with col3:
        st.metric("높은 위험", f"{high_risk:,}명", f"{(high_risk/total_customers)*100:.1f}%")
    
    with col4:
        st.metric("중간 위험", f"{medium_risk:,}명", f"{(medium_risk/total_customers)*100:.1f}%")
    
    with col5:
        st.metric("낮은 위험", f"{low_risk:,}명", f"{(low_risk/total_customers)*100:.1f}%")

def display_risk_customers(results_df, display_columns):
    st.markdown("### 위험도별 고객 목록")
    
    tab1, tab2, tab3 = st.tabs(["🔴 높은 위험", "🟡 중간 위험", "🟢 낮은 위험"])
    
    def style_dataframe(df):
        def highlight_risk(val):
            try:
                prob = float(val.strip('%')) / 100
                if prob >= 0.7:
                    return 'background-color: #ffcccc'
                elif prob >= 0.4:
                    return 'background-color: #fff2cc'
                else:
                    return 'background-color: #d9ead3'
            except:
                return ''
        
        return df.style.apply(lambda x: [''] * len(x) if x.name != '이탈 확률' 
                            else [highlight_risk(v) for v in x], axis=0)\
                    .set_properties(**{
                        'text-align': 'left',
                        'white-space': 'pre-wrap',
                        'font-size': '14px',
                        'padding': '10px'
                    })\
                    .set_table_styles([
                        {'selector': 'th',
                         'props': [('font-size', '14px'),
                                  ('text-align', 'left'),
                                  ('padding', '10px'),
                                  ('white-space', 'pre-wrap')]},
                        {'selector': 'td',
                         'props': [('min-width', '100px')]}
                    ])
    
    with tab1:
        high_risk_df = results_df[results_df['이탈 확률'] >= 0.7].copy()
        if not high_risk_df.empty:
            high_risk_df['이탈 확률'] = high_risk_df['이탈 확률'].apply(lambda x: f"{x:.1%}")
            st.dataframe(style_dataframe(high_risk_df[display_columns].sort_values('이탈 확률', ascending=False)),
                         height=400, use_container_width=True)
        else:
            st.info("높은 위험군에 해당하는 고객이 없습니다.")
    
    with tab2:
        medium_risk_df = results_df[(results_df['이탈 확률'] >= 0.4) & 
                                    (results_df['이탈 확률'] < 0.7)].copy()
        if not medium_risk_df.empty:
            medium_risk_df['이탈 확률'] = medium_risk_df['이탈 확률'].apply(lambda x: f"{x:.1%}")
            st.dataframe(style_dataframe(medium_risk_df[display_columns].sort_values('이탈 확률', ascending=False)),
                         height=400, use_container_width=True)
        else:
            st.info("중간 위험군에 해당하는 고객이 없습니다.")
    
    with tab3:
        low_risk_df = results_df[results_df['이탈 확률'] < 0.4].copy()
        if not low_risk_df.empty:
            low_risk_df['이탈 확률'] = low_risk_df['이탈 확률'].apply(lambda x: f"{x:.1%}")
            st.dataframe(style_dataframe(low_risk_df[display_columns].sort_values('이탈 확률', ascending=False)),
                         height=400, use_container_width=True)
        else:
            st.info("낮은 위험군에 해당하는 고객이 없습니다.")

def calculate_risk_info(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    numeric_info = df[numeric_cols].mean().to_dict()
    categorical_info = df[categorical_cols].apply(lambda x: x.value_counts().to_dict()).to_dict()
    
    return {**numeric_info, **categorical_info}