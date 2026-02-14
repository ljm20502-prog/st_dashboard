import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from collect_naver_data import NaverDataCollector
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# 페이지 설정
st.set_page_config(page_title="Naver API 데이터 분석 대시보드", layout="wide")
st.title("📊 Naver API 실시간 데이터 분석 대시보드")

# 사이드바 설정
st.sidebar.header("🔍 검색 및 설정")
keyword1 = st.sidebar.text_input("첫 번째 키워드", value="오메가3")
keyword2 = st.sidebar.text_input("두 번째 키워드", value="비타민D")
search_btn = st.sidebar.button("데이터 수집 및 분석 시작")

# 카테고리 ID (기본값: 건강식품 50000008)
CAT_ID = "50000008"

# 세션 상태 초기화
if "data" not in st.session_state:
    st.session_state.data = None

def get_full_data(kw):
    """키워드에 대한 모든 데이터 수집 및 분석"""
    collector = NaverDataCollector(CLIENT_ID, CLIENT_SECRET)
    
    # 1. 쇼핑 트랜드
    url_trend = "https://openapi.naver.com/v1/datalab/shopping/categories"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    body = {
        "startDate": start_date, "endDate": end_date, "timeUnit": "date",
        "category": [{"name": kw, "param": [CAT_ID]}]
    }
    import requests, json
    res_tr = requests.post(url_trend, headers=collector.headers, data=json.dumps(body))
    df_trend = pd.DataFrame(res_tr.json()['results'][0]['data']) if res_tr.status_code == 200 else pd.DataFrame()
    
    # 2. 블로그 검색
    url_blog = f"https://openapi.naver.com/v1/search/blog.json?query={kw}&display=100"
    res_blog = requests.get(url_blog, headers=collector.headers)
    df_blog = pd.DataFrame(res_blog.json()['items']) if res_blog.status_code == 200 else pd.DataFrame()
    
    # 3. 쇼핑 검색
    url_shop = f"https://openapi.naver.com/v1/search/shop.json?query={kw}&display=100"
    res_shop = requests.get(url_shop, headers=collector.headers)
    df_shop = pd.DataFrame(res_shop.json()['items']) if res_shop.status_code == 200 else pd.DataFrame()
    df_shop['lprice'] = pd.to_numeric(df_shop['lprice'], errors='coerce')
    
    return {"trend": df_trend, "blog": df_blog, "shop": df_shop}

if search_btn:
    with st.spinner("네이버 API에서 데이터를 수집 중입니다..."):
        data1 = get_full_data(keyword1)
        data2 = get_full_data(keyword2)
        st.session_state.data = {keyword1: data1, keyword2: data2}
        st.success("데이터 수집 완료!")

if st.session_state.data:
    kw1, kw2 = keyword1, keyword2
    d1, d2 = st.session_state.data[kw1], st.session_state.data[kw2]
    
    tab1, tab2, tab3 = st.tabs(["🚀 트랜드 비교", "📝 블로그 분석", "🛒 쇼핑 마켓"])
    
    with tab1:
        st.subheader("연간 쇼핑 클릭 트랜드")
        if not d1['trend'].empty and not d2['trend'].empty:
            fig_tr = go.Figure()
            fig_tr.add_trace(go.Scatter(x=d1['trend']['period'], y=d1['trend']['ratio'], name=kw1))
            fig_tr.add_trace(go.Scatter(x=d2['trend']['period'], y=d2['trend']['ratio'], name=kw2))
            fig_tr.update_layout(title="키워드별 연간 클릭 추이", xaxis_title="날짜", yaxis_title="클릭 비율")
            st.plotly_chart(fig_tr, use_container_width=True)
            
            # 표 1: 트랜드 기술통계
            st.write("### 트랜드 요약 통계")
            col1, col2 = st.columns(2)
            col1.write(f"**{kw1}**")
            col1.dataframe(d1['trend'].describe().T)
            col2.write(f"**{kw2}**")
            col2.dataframe(d2['trend'].describe().T)
        else:
            st.warning("트랜드 데이터를 불러올 수 없습니다.")

    with tab2:
        st.subheader("블로그 텍스트 마이닝 (TF-IDF)")
        def plot_tfidf(df, kw, color):
            corpus = (df['title'] + " " + df['description']).fillna("")
            tfidf = TfidfVectorizer(max_features=20)
            matrix = tfidf.fit_transform(corpus)
            freq = pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out()).sum().sort_values(ascending=False)
            fig = px.bar(freq, orientation='h', title=f"{kw} 블로그 핵심 키워드", color_continuous_scale=color)
            return fig, freq

        c1, c2 = st.columns(2)
        fig1, freq1 = plot_tfidf(d1['blog'], kw1, 'Blues')
        fig2, freq2 = plot_tfidf(d2['blog'], kw2, 'Reds')
        c1.plotly_chart(fig1, use_container_width=True)
        c2.plotly_chart(fig2, use_container_width=True)
        
        # 표 2: TF-IDF 키워드 순위
        st.write("### 키워드 가중치 TOP 20")
        col1, col2 = st.columns(2)
        col1.dataframe(freq1.rename("가중치").head(20))
        col2.dataframe(freq2.rename("가중치").head(20))

    with tab3:
        st.subheader("쇼핑 시장 분석")
        
        # 그래프 3: 가격 분포 히스토그램
        fig_price = go.Figure()
        fig_price.add_trace(go.Histogram(x=d1['shop']['lprice'], name=kw1, opacity=0.75))
        fig_price.add_trace(go.Histogram(x=d2['shop']['lprice'], name=kw2, opacity=0.75))
        fig_price.update_layout(barmode='overlay', title="상품 가격 분포 비교")
        st.plotly_chart(fig_price, use_container_width=True)
        
        # 그래프 4 & 5: 브랜드 점유율 (Pie) & 가격 분포 (Box)
        c1, c2 = st.columns(2)
        brand_counts = d1['shop']['brand'].value_counts().head(10)
        fig_pie = px.pie(values=brand_counts.values, names=brand_counts.index, title=f"{kw1} 브랜드 점유율 (Top 10)")
        c1.plotly_chart(fig_pie, use_container_width=True)
        
        fig_box = px.box(d1['shop'], x="brand", y="lprice", title=f"{kw1} 브랜드별 가격 분포", 
                         category_orders={"brand": brand_counts.index.tolist()})
        c2.plotly_chart(fig_box, use_container_width=True)
        
        # 표 3, 4, 5
        st.write("### 쇼핑 상세 분석 표")
        t1, t2, t3 = st.columns(3)
        t1.write("**브랜드별 평균 가격**")
        t1.dataframe(d1['shop'].groupby('brand')['lprice'].mean().sort_values(ascending=False).head(10))
        
        t2.write("**판매몰별 상품 수**")
        t2.dataframe(d1['shop']['mallName'].value_counts().head(10))
        
        t3.write("**원본 데이터 샘플 (최근 5건)**")
        t3.dataframe(d1['shop'][['title', 'lprice', 'mallName']].head(5))

else:
    st.info("사이드바에서 키워드를 입력하고 버튼을 눌러 분석을 시작하세요.")

st.markdown("---")
st.caption("Produced by Antigravity AI Agent | Data from Naver Open API")
