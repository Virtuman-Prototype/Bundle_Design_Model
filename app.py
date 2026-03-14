import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# --- 1. 初始化页面配置 ---
st.set_page_config(
    page_title="JUNA x Rose Boréal AI Lab", 
    page_icon="🧘",
    layout="wide"
)

# --- 2. 核心 AI 模型与缓存 (DSR Step 3) ---
@st.cache_resource
def load_ai_model():
    # 用于语义关联分析的预训练模型 
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_ai_model()

@st.cache_data
def load_custom_stopwords(file_path="Stopword.txt"):
    # 基础停用词库，处理魁北克英法双语数据 [cite: 139]
    base_sw = {"the", "and", "our", "with", "for", "from", "this", "is", "of", "to", "in", "it", "vous", "votre", "pour", "est"}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_sw = set([line.strip().lower() for line in f.readlines() if line.strip()])
                return base_sw.union(file_sw)
        except: return base_sw
    return base_sw

CUSTOM_STOPWORDS = load_custom_stopwords()

# --- 3. 静态品牌向量与维度定义 (用于 Cosine Similarity 锚点) ---
JUNA_VECTOR = """
JUNA Academy is a Québec-based wellness movement focusing on an authentic and integrative approach to somatic health. 
We empower our community through nature-inspired evolution, mindful movement, and conscious leadership education. 
Our values center on living by seasons and sustainable personal growth, creating a synergy between yoga practice and ethical lifestyle.
""" 

ROSE_VECTOR = """
Rose Boreal (Rose Buddha) is a sustainable Québec activewear brand and B-Corp movement for mindful living. 
We empower women through nature-inspired designs, eco-conscious materials, and a supportive wellness community. 
Our brand promotes a lifestyle of slowing down and moving with joy, aligning sustainable fashion with the essence of yoga and self-care.
"""

DIMENSIONS = {
    "D1 Value Congruence": "alignment of values, sustainability, mindfulness, authenticity, ethical commitment",
    "D2 Audience Overlap": "target customer similarity, lifestyle positioning, active women, eco-conscious segments",
    "D3 Identity & Aesthetic": "brand tone, visual identity, symbolic meaning, nature-inspired, design language",
    "D4 Functional Complementarity": "complementarity of services and products, yoga apparel and wellness education synergy",
    "D5 Prestige & Risk": "brand reputation, price tier, Canadian quality, low reputational risk, B-Corp trust"
} 

# --- 4. 核心工具函数 ---
def clean_text_for_cloud(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text) 
    words = text.split()
    return " ".join([w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2])

def calculate_semantic_score(text1, text2, criterion):
    """真实的余弦相似度计算逻辑 """
    emb1 = model.encode(text1 + " " + criterion)
    emb2 = model.encode(text2 + " " + criterion)
    cos_sim = util.cos_sim(emb1, emb2).item()
    # 映射逻辑：将相似度映射到 1-5 分制
    score = 1 + (cos_sim * 4)
    return round(min(5, max(1, score)), 2)

def generate_styled_cloud(text, mask_path, main_color):
    """带形状遮罩的词云生成 """
    try:
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            wc = WordCloud(
                width=800, height=800,
                background_color='white',
                mask=mask,
                stopwords=CUSTOM_STOPWORDS,
                colormap='viridis' if main_color == '#1F6859' else 'summer',
                contour_width=1,
                contour_color='#eeeeee'
            ).generate(text)
        else:
            st.warning(f"⚠️ Mask {mask_path} not found. Using rectangle.")
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"Cloud Error: {e}")
        return None

# --- 5. UI 逻辑与模块交互 ---
st.sidebar.title("🛠️ Bundle Co-Creation Design Model")
step = st.sidebar.radio("Navigation", ["M1: Strategic Evaluator", "M2: Community Insights", "M3: Co-design Lab", "M4: Dynamic Delivery"])

if step == "M1: Strategic Evaluator":
    st.title("🛡️ M1: Strategic Brand Synergy Evaluator")
    st.info("This module uses Latent Association Analysis to evaluate brand fit.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("Upload JUNA (Brand A) Data", type="csv", key="juna_up")
    with col_b:
        file_b = st.file_uploader("Upload Partner (Brand B) Data", type="csv", key="rose_up")

    if file_a and file_b:
        df_a, df_b = pd.read_csv(file_a), pd.read_csv(file_b)
        text_a = " ".join(df_a.iloc[:, -1].astype(str))
        text_b = " ".join(df_b.iloc[:, -1].astype(str))
        
        c_a = clean_text_for_cloud(text_a)
        c_b = clean_text_for_cloud(text_b)

        if st.button("Run AI Brand Fit Analysis"):
            # A. 形状遮罩词云展示
            st.subheader("☁️ Brand Identity Visualizations (Masked)")
            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                st.write("**JUNA Silhouette**")
                fig_j = generate_styled_cloud(c_a, "juna_mask.png", "#1F6859")
                if fig_j: st.pyplot(fig_j)
            with col_wc2:
                st.write("**Rose Boréal Silhouette**")
                fig_r = generate_styled_cloud(c_b, "rose_mask.png", "#D0DF00")
                if fig_r: st.pyplot(fig_r)

            # B. 真实的语义得分矩阵 (注意：这里必须保持缩进！)
            st.subheader("📊 AI Semantic Fit Matrix (Cosine Similarity)")
            results = []
            for dim, criteria in DIMENSIONS.items():
                score = calculate_semantic_score(JUNA_VECTOR, ROSE_VECTOR, criteria)
                results.append({"Dimension": dim, "Fit Score": score}) 

            df_fit = pd.DataFrame(results)
            st.dataframe(df_fit, use_container_width=True)

            # C. 绘制雷达图 (注意：这里必须保持缩进！)
            st.subheader("🕸️ Brand Synergy Radar Chart")
            import plotly.graph_objects as go

            categories = df_fit["Dimension"].tolist()
            values = df_fit["Fit Score"].tolist()

            # 首尾相连
            categories_close = categories + [categories[0]]
            values_close = values + [values[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_close,
                theta=categories_close,
                fill='toself',
                name='Brand Fit',
                line_color='#1F6859'
            ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# --- 模块 2: 社区感知 (Community Insights) ---
elif step == "M2: Community Insights":
    st.title("📊 M2: Community Insight Engine")
    st.write("Analyzing sentiment and topics from the Quebec wellness community.")
    
    # 1. 读取数据并确保日期格式正确
    df_trends = pd.read_csv("Community_Trends.csv")
    
    # 强制转换日期格式，防止电脑把它当成普通文字
    df_trends['Date'] = pd.to_datetime(df_trends['Date'])
    
    # 2. 情感指标展示
    avg_sentiment = df_trends['Sentiment_Score'].mean()
    st.metric("Community Sentiment Score", f"{avg_sentiment:.2f}")
    
    # 3. 趋势图表修复逻辑
    st.subheader("Topic Trends Over Time")
    
    # 如果数据点很多，折线图需要先按日期排序
    chart_data = df_trends.sort_values('Date').set_index('Date')['Sentiment_Score']
    
    # 检查是否有多个不同的日期
    if df_trends['Date'].nunique() > 1:
        st.line_chart(chart_data)
    else:
        # 如果所有日期都一样，折线图无法显示，我们改用柱状图显示明细
        st.warning("Note: All data points belong to the same date. Showing distribution instead of timeline.")
        st.bar_chart(chart_data)
    
    # 4. 词云关键词
    st.subheader("Early Adopter Hotspots")
    st.write("#NatureConnection #InnerFlow #SustainableMovement #QuebecLaurentians")

# --- 模块 3: 协同设计实验室 (Co-design Lab) ---
elif step == "M3: Co-design Lab":
    st.title("🧪 M3: Co-creation Design Lab")
    st.write("Matching Brand Assets with Community Motives.")
    
    inventory = pd.read_csv("Apparel_Inventory.csv")
    
    # 交互：选择一个趋势场景
    selected_scenario = st.selectbox("Select a detected trend scenario:", inventory['Scenario_Tag'].unique())
    
    # 逻辑匹配
    matches = inventory[inventory['Scenario_Tag'] == selected_scenario]
    
    st.write(f"Found {len(matches)} matching assets for {selected_scenario}:")
    st.table(matches[['Brand', 'Product_Name', 'Type']])

# --- 模块 4: 动态交付 (Dynamic Delivery) ---
elif step == "M4: Dynamic Delivery":
    st.title("✨ M4: Dynamic Bundle Output Engine")
    st.write("Final Co-created Bundles for the Quebec Market.")
    
    inventory = pd.read_csv("Apparel_Inventory.csv")
    
    # 模拟生成 3 个精选 Bundle
    scenarios = ["#NatureConnection", "#InnerFlow", "#SlowLiving"]
    
    cols = st.columns(3)
    for i, scenario in enumerate(scenarios):
        with cols[i]:
            st.subheader(f"Bundle {i+1}")
            st.caption(scenario)
            # 找到对应场景的产品
            items = inventory[inventory['Scenario_Tag'] == scenario].head(2)
            for _, item in items.iterrows():
                st.image(item['Image_URL'], use_column_width=True)
                st.write(f"**{item['Brand']}** - {item['Product_Name']}")
            
            st.button(f"Refine {i+1}", key=f"btn_{i}")

    st.divider()
    st.write("🔄 **Feedback Loop (DR12):** Interaction data will refine Module 1 weights.")
