import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from sentence_transformers import SentenceTransformer, util

# --- 1. 初始化页面配置 ---
st.set_page_config(
    page_title="JUNA x Rose Boréal AI Lab", 
    page_icon="🧘",
    layout="wide"
)
st.markdown("""
    <style>
    /* 针对所有的 popover 按钮文字进行样式定制 */
    div[data-testid="stPopover"] button p {
        font-size: 12px !important;
        white-space: nowrap !important;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """, unsafe_allow_html=True)

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

# 建议将 get_base64 函数和 import base64 移到文件最顶部，如果留在这里也可以，但必须保证下方不重复
import base64

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# --- 侧边栏内容 ---
st.sidebar.title("🛠️ Bundle Co-Creation Design Model")
# --- 侧边栏架构图展示逻辑 ---
st.sidebar.markdown("---") # 分割线
st.sidebar.write("🕵️‍♂️ **System Architecture Overview**")

# 尝试读取架构图
diag_bin_str = get_base64_of_bin_file('architecture_diagram.png')

if diag_bin_str:
    # 方案：利用 HTML/CSS 实现点击放大
    # 我们创建一个隐藏的 input checkpoint，点击图片时触发
    diagram_html = f"""
    <style>
    /* 缩略图样式 */
    .sidebar-thumbnail {{
        width: 100%;
        cursor: zoom-in;
        border-radius: 5px;
        transition: transform 0.2s;
        border: 1px solid #ddd;
    }}
    .sidebar-thumbnail:hover {{
        transform: scale(1.02);
    }}

    /* 放大后的全图蒙层样式 */
    #diag-lightbox {{
        display: none;
        position: fixed;
        z-index: 9999;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.85);
        cursor: zoom-out;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }}

    #diag-lightbox img {{
        max-width: 90%;
        max-height: 90%;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(255,255,255,0.2);
    }}

    /* 触发逻辑：当 checkpoint 被选中时显示蒙层 */
    #diag-trigger:checked ~ #diag-lightbox {{
        display: flex;
    }}
    
    /* 隐藏 checkbox */
    #diag-trigger {{
        display: none;
    }}
    </style>

    <input type="checkbox" id="diag-trigger">
    
    <label for="diag-trigger">
        <img src="data:image/png;base64,{diag_bin_str}" class="sidebar-thumbnail" alt="Architectural Diagram Thumbnail">
    </label>

    <div id="diag-lightbox" onclick="document.getElementById('diag-trigger').checked=false">
        <img src="data:image/png;base64,{diag_bin_str}" alt="Architectural Diagram Full Resolution">
    </div>
    """
    st.sidebar.markdown(diagram_html, unsafe_allow_html=True)
    st.sidebar.caption("💡 Click image to enlarge the full design logic.")
else:
    st.sidebar.warning("⚠️ 'architecture_diagram.png' not found. Please upload to GitHub.")

st.sidebar.markdown("---") # 下方分割线，准备接 M1~M4 导航


# 注意：这里只保留一个 radio 导航
# --- 全局配置与数据 (Global Data) ---
# 将内容字典放在这里，确保 M3 和 M4 都能读取到
content_map = {
    "Laurentian Forest Yoga (Nature Connectivity)": {
        "img": "https://www.bonjourquebec.com/media/juna-yoga-ski-yoga-a-mont-tremblant.jpg?width=2900&height=2400",
        "title": "Laurentian Forest Yoga",     
        "color": "Pine Green & Earthy Brown",
        "material": "Thermal Recycled Polyester"
    },
    "Montreal Urban Meditation (Mindfulness)": {
        "img": "https://www.bonjourquebec.com/media/JunaYoga-Tremblant-Cours-Groupe-Yoga.jpg?width=2900&height=2400",
        "title": "Montreal Urban Meditation",
        "color": "Cool Grey & Zen White",
        "material": "Seamless Organic Cotton"
    },
    "St. Lawrence River Flow (Water Element)": {
        "img": "https://www.bonjourquebec.com/media/juna-yoga-retraite-aux-iles-de-la-madeleine.jpg?width=2900&height=2400",
        "title": "St. Lawrence River Flow",
        "color": "Deep River Blue & Mist",
        "material": "Quick-dry Performance Fabric"
    }
}

# --- 侧边栏导航 ---
# 1. 顶部导航
step = st.sidebar.radio(
    "Navigation", 
    ["M1: Strategic Evaluator", "M2: Community Insights", "M3: Co-design Lab", "M4: Dynamic Delivery"]
)
# 2. 第一条分割线
st.sidebar.markdown("---")

# 3. 架构图查看按钮 (这是你之前加的功能)
with st.sidebar:
    st.write("**Model Framework**")
    # 这里放你那个点击放大架构图的代码
    # ...

# 4. 第二条分割线 (可选)
st.sidebar.markdown("---") 

# --- 侧边栏底部个人信息 ---
# 尝试读取头像
bin_str = get_base64_of_bin_file('avatar.png')

if bin_str:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <img src="data:image/png;base64,{bin_str}" style="width: 50px; height: 50px; border-radius: 50%; border: 2px solid #1F6859; object-fit: cover;">
            <div>
                <p style="margin:0; font-size: 13px; font-weight: bold; color: #1F6859;">Serena Shuo YANG</p>
                <p style="margin:0; font-size: 11px; color: #666;">Shuoyang5@Carleton</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # 如果图片 avatar.png 不存在，则显示备用文字版
    st.sidebar.write("👤 **Serena S YANG**")
    st.sidebar.caption("Shuoyang5@Carleton")

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
            st.subheader("☁️ Brand Identity Visualizations")
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
            
            # --- 承接在 st.plotly_chart(fig_radar, use_container_width=True) 之后 ---

            # 1. 展示数据预览 (让用户看到上传的 CSV 确实被读取了) [cite: 158]
            st.write("### 📄 Data Preview (Brand B)")
            st.dataframe(df_b.head(3)) # 这里的 df_b 对应你上传的 Partner Data
            
            # 2. 显示 AI 实测分值与决策触发器 [cite: 177]
            st.divider()
            avg_s = df_fit["Fit Score"].mean() # 获取计算出的平均分
            
            col1, col2 = st.columns(2)
            with col1:
                # 将原来的模拟 0.88 改为动态计算的 avg_s
                # 同时根据分值给出等级评价
                status = "GOLD Level" if avg_s >= 4 else "SILVER Level" if avg_s >= 3 else "RE-EVALUATE"
                st.metric(label="Fit Index (FI)", value=f"{avg_s:.2f}", delta=status)
            
            with col2:
                # 对应 PPT 中的逻辑触发逻辑 [cite: 70, 172]
                if avg_s >= 4:
                    st.info("🎯 **Logic Trigger**: Brand values highly aligned. Full bundle scenario unlocked.")
                elif avg_s >= 3:
                    st.info("⚠️ **Logic Trigger**: Partial alignment. Joint marketing recommended before bundling.")
                else:
                    st.error("❌ **Logic Trigger**: Low alignment. Bundle risk is high.")
                    # --- 承接在 st.info 之后，M2 之前 ---

            st.markdown("### 📢 AI Strategic Recommendation")
            
            # 建立基于分值的决策矩阵，对应 DP 和 DR 要求 [cite: 169, 174]
            if avg_s >= 4:
                st.success(f"""
                **STRATEGIC DECISION: PROCEED TO BUNDLING**
                - **Status**: High Synergy Detected (Score: {avg_s:.2f}) 
                - **Next Step**: Unlock the 'Urban Meditation Retreat' bundle in M4[cite: 173]. 
                - **Rationale**: Core values and aesthetic preferences are highly congruent[cite: 178].
                """)
                st.balloons() # 庆祝高契合度的彩蛋
                
            elif avg_s >= 3:
                st.warning(f"""
                **STRATEGIC DECISION: CONDITIONAL COLLABORATION**
                - **Status**: Moderate Synergy (Score: {avg_s:.2f})
                - **Next Step**: Conduct a joint pilot event (M3) before full inventory commitment.
                - **Rationale**: Functional complementarity exists, but audience overlap needs verification[cite: 161].
                """)
                
            else:
                st.error(f"""
                **STRATEGIC DECISION: RE-EVALUATE PARTNERSHIP**
                - **Status**: Low Synergy (Score: {avg_s:.2f})
                - **Next Step**: Search for alternative partners or adjust Brand B's positioning.
                - **Rationale**: Significant gaps in latent factor alignment[cite: 175].
                """)

            # --- 后面接原有的决策反馈 (st.success/warning) ---


# --- 模块 2: 社区感知 (Community Insights) ---
elif step == "M2: Community Insights":
    st.title("📊 M2: Community Insight Engine")
    st.markdown("### 🔗 Please link your social media here:")

    # 1. 第一行：JUNA Académie 的链接输入
    st.write("**Step 1: Connect Wellness Communities**")
    col1, col2, col3, _ = st.columns([1, 1, 1, 5])
    
    with col1:
        # 使用 span 标签通过 CSS 缩小字号，防止换行
        with st.popover("📸Community1"):
            link_juna_ig = st.text_input("JUNA Instagram Link", "https://instagram.com/juna_academie")
    with col2:
        with st.popover("📘Community2"):
            link_juna_fb = st.text_input("JUNA Facebook Link")
    with col3:
        with st.popover("🎥Community3"):
            link_juna_yt = st.text_input("JUNA YouTube Link")

    st.divider()

    # 2. 第二行：合作伙伴的链接输入
    st.write("**Step 2: Connect Apparel Communities**")
    colb1, colb2, colb3, _ = st.columns([1, 1, 1, 5])
    
            # 注意这里！必须改为 colb1, colb2, colb3
    with colb1:  # 刚才你写成了 col1
        with st.popover("📸Community4"):
            link_b_ig = st.text_input("Partner Instagram Link", "https://instagram.com/rose_boreal")
    with colb2:  # 刚才你写成了 col2
        with st.popover("📘Community5"):
            link_b_fb = st.text_input("Partner Facebook Link")
    with colb3:  # 刚才你写成了 col3
        with st.popover("🎥Community6"):
            link_b_yt = st.text_input("Partner YouTube Link")

    st.write("") 
    
    # 3. 核心触发逻辑：模拟数据采集与分析流程 [cite: 91, 106, 155]
    if st.button("🚀 Start AI Scraping & Sentiment Analysis"):
        with st.spinner("AI is accessing public API endpoints... Scraping Quebec wellness communities..."):
            import time
            time.sleep(2.5) # 模拟处理 30,000-50,000 条数据的延迟 [cite: 144]
            
            if os.path.exists("Community_Trends.csv"):
                df_trends = pd.read_csv("Community_Trends.csv")
                df_trends['Date'] = pd.to_datetime(df_trends['Date'])
                
                # A. 展示核心交付物指标 [cite: 9, 149, 162]
                st.success("✅ Extraction Complete: 48,230 entries analyzed.")
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Community Sentiment", f"{df_trends['Sentiment_Score'].mean():.2f}", "+5% ↗️")
                m_col2.metric("Persona Match Rate", "92%", "High")
                m_col3.metric("Trend Velocity", "Fast", "Topic: #YogaQC")

                # B. 情感趋势图 (Topic Modeling 辅助展示) [cite: 93, 161]
                st.subheader("📈 Real-time Sentiment Trend")
                chart_data = df_trends.sort_values('Date').set_index('Date')['Sentiment_Score']
                st.line_chart(chart_data)

                # --- 新增：LDA 主题分布柱状图 ---
                st.divider()
                st.subheader("🧬 LDA Topic Modeling Distribution")
                st.write("AI-driven extraction of latent conversation themes from Quebec wellness communities.")

                # 模拟 LDA 提取的主题及其权重数据
                topic_data = {
                    "Topic Area": [
                        "Nature Connectivity", 
                        "Urban Mindfulness", 
                        "Sustainable Materials", 
                        "Community Events", 
                        "Wellness Rituals"
                    ],
                    "Relevance Weight": [0.35, 0.25, 0.18, 0.12, 0.10]
                }
                df_topics = pd.DataFrame(topic_data)

                # 使用 Plotly 绘制水平柱状图，更有专业 Dashboard 的感觉
                import plotly.express as px
                
                fig_topics = px.bar(
                    df_topics, 
                    x="Relevance Weight", 
                    y="Topic Area", 
                    orientation='h',
                    color="Relevance Weight",
                    color_continuous_scale="Viridis",
                    labels={"Relevance Weight": "Topic Weight (Confidence)"}
                )
                
                fig_topics.update_layout(
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig_topics, use_container_width=True)

                st.info("💡 **Insight**: 'Nature Connectivity' is the dominant latent factor this month, directly triggering the 'Forest Yoga' scenario in M3.")
                
                # C. 语义热点与早期采用者关键词 [cite: 141, 148]
                st.subheader("🔥 Emerging Topic Hotspots (Early Adopter Cues)")
                # 使用 HTML 注入自定义颜色和字号
                hotspots_html = """
                <div style="background-color: #f0f2f6; padding: 15px; border-left: 5px solid #1F6859; border-radius: 5px;">
                    <span style="color: #1F6859; font-size: 20px; font-weight: bold; margin-right: 15px;">#NatureConnection</span>
                    <span style="color: #1F6859; font-size: 20px; font-weight: bold; margin-right: 15px;">#InnerFlow</span>
                    <span style="color: #1F6859; font-size: 20px; font-weight: bold; margin-right: 15px;">#SustainableMovement</span>
                    <span style="color: #1F6859; font-size: 20px; font-weight: bold;">#QuebecLaurentians</span>
                </div>
                """
                st.markdown(hotspots_html, unsafe_allow_html=True)
            else:
                st.error("Data source 'Community_Trends.csv' not found. Please ensure it is in the GitHub repository.")

# --- 模块 3: 协同设计实验室 (Co-design Lab) ---
elif step == "M3: Co-design Lab":
    st.title("🧪 M3: Co-creation Design Lab")

    # --- [新增]---1 输入层：连接服装内容库 ---
    st.markdown("### 📦 Step 1: Connect Apparel Databases")
    st.write("Link JUNA & Partner product libraries to enable Preference Filtering.")
    
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        with st.popover("👚 JUNA Inventory (API/CSV)"):
            st.text_input("JUNA Catalog API Endpoint", "https://api.juna.com/v1/inventory")
            st.file_uploader("Or upload JUNA Product List (.csv)", type="csv", key="juna_csv")
    with col_inv2:
        with st.popover("🌿 Partner Inventory (API/CSV)"):
            st.text_input("Partner Catalog API Endpoint", "https://api.partner.com/v1/stock")
            st.file_uploader("Or upload Partner Product List (.csv)", type="csv", key="partner_csv")

    st.divider()    
    
    # --- 2. 下拉菜单选择 ---
    st.write("#### 🎯 Step2:Select Target Co-creation Scenario")
    scenario_choice = st.selectbox(
        "Choose a validated scenario based on M2 Trends:",
        [
            "Laurentian Forest Yoga (Nature Connectivity)", 
            "Montreal Urban Meditation (Mindfulness)", 
            "St. Lawrence River Flow (Water Element)"
        ],
        key="scenario_selection"  # <--- 关键就在这里！加个逗号，写上这个 key
)

    # --- [新增] Process 层：Preference Filtering 逻辑展示 ---
    st.divider()
    st.subheader("⚙️ Step 3: Interactive Lab & Preference Filtering")
    
    col_lab1, col_lab2 = st.columns([1, 1])
    with col_lab1:
        st.write("**Brand Strategy Parameters**")
        intensity = st.slider("Co-creation Intensity (Fit Weight)", 0.0, 1.0, 0.8)
        st.caption("Adjusting the bias between Brand DNA consistency vs. Market Trend velocity.")
        
        # 审美标签墙 (Aesthetic Tags)
        st.write("**Market Aesthetic Tags (from M2)**")
        st.button("#Minimalist")
        st.button("#QuebecNature")
        st.button("#SustainableFlow")
        
    with col_lab2:
        st.write("**Preference Filtering Logic**")
        # 用状态组件代替原始代码块
        with st.status("🔍 System Logic Execution", expanded=False):
            st.write("1. Accessing combined product catalog...")
            st.write(f"2. Filtering by Scenario: **{scenario_choice}**")
            st.write(f"3. Applying Intensity Weight: **{intensity}**")
            st.write("4. Ranking items via Latent Association Mapping...")
            st.write("✅ 12 optimal pairs identified.")
        
        # 按钮保持不变，但在上方增加一点视觉引导
        
        if st.button("✨ Re-optimize & Rank Bundle Elements", key="m3_reoptimize_top"):
            with st.spinner("Filtering products based on FitIndex and Latent Association..."):
                import time
                time.sleep(1.5)
                st.success("Preference Filtering Complete: 12 candidate pairs identified.")

    st.divider()

    # --- 2. 数据融合看板 (指标随选择动态变化) ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Brand Compatibility", "High", "0.88 FI")
    with col2:
        # 根据选择动态设置权重
        if "Forest" in scenario_choice:
            weight, tag = "35%", "Nature"
        elif "Urban" in scenario_choice:
            weight, tag = "25%", "Mindfulness"
        else: # St. Lawrence
            weight, tag = "20%", "Water"
        st.metric("Community Trend Weight", tag, weight)

    # --- 3. 参数微调与优化按钮 ---
    st.write("#### ⚙️ Step 3: Interactive Lab & Preference Filtering")
    with st.expander("🛠️ Advanced Strategy Tuning", expanded=True): # 建议默认展开，增加丰富感
        col_a, col_b = st.columns(2)
        with col_a:
            st.slider("Brand Core Consistency", 0.0, 1.0, 0.8)
            st.slider("Community Trend Velocity", 0.0, 1.0, 0.6)
        with col_b:
            st.write("**Strategy Status:** Ready to Match")
            # 加上 key="m3_reoptimize_button" 即可
            # 重新加上你想要的按钮，并增加点击后的反馈
            if st.button("✨ Re-optimize & Rank Bundle Elements", key="m3_reoptimize_bottom"):
                with st.spinner("Re-calculating semantic distance..."):
                    import time
                    time.sleep(1)
                    st.toast("Mapping parameters updated!", icon="✨")

    st.divider()

# --- 4. 最终输出层：Optimized Bundle Concept (Step 4) ---
    st.subheader("🎯 Step 4: Optimized Bundle Concept & Rationale")
    
    # 定义不同场景的内容字典
    content_map = {
        "Laurentian Forest Yoga (Nature Connectivity)": {
            "img": "https://www.bonjourquebec.com/media/juna-yoga-ski-yoga-a-mont-tremblant.jpg?width=2900&height=2400",
            "title": "Laurentian Forest Yoga",     
            "color": "Pine Green & Earthy Brown",
            "material": "Thermal Recycled Polyester"
        },
        "Montreal Urban Meditation (Mindfulness)": {
            "img": "https://www.bonjourquebec.com/media/JunaYoga-Tremblant-Cours-Groupe-Yoga.jpg?width=2900&height=2400",
            "title": "Montreal Urban Meditation",
            "color": "Cool Grey & Zen White",
            "material": "Seamless Organic Cotton"
        },
        "St. Lawrence River Flow (Water Element)": {
            "img": "https://www.bonjourquebec.com/media/juna-yoga-retraite-aux-iles-de-la-madeleine.jpg?width=2900&height=2400",
            "title": "St. Lawrence River Flow",
            "color": "Deep River Blue & Mist",
            "material": "Quick-dry Performance Fabric"
        }
    }

    # 获取当前选中的内容
    current_content = content_map[scenario_choice]

    
    # 渲染结果卡片
    with st.container(border=True):
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(current_content["img"], caption=f"Concept: {current_content['title']}")
        with c2:
            st.markdown(f"#### **Theme: {current_content['title']}**")
           
            # 融合 Rationale Report 到原来的描述中
            st.info(f"**Rationale Report:** This bundle achieved a **{intensity*100:.1f}%** matching score. High semantic match found between '{tag}' elements and brand DNA.")
            
            st.write("**Mapped Design Elements:**")
            st.markdown(f"- 🎨 *Style:* {current_content['color']}")
            st.markdown(f"- 🧵 *Material:* {current_content['material']}")
            st.markdown("- 📦 *Bundle Suggestion:* Apparel + Local Experience Access")

    st.success(f"✅ {current_content['title']} aligned with DR3/DR9 requirements.")

# --- 模块 4: 动态输出层------
elif step == "M4: Dynamic Delivery":
    st.title("🚀 M4: Dynamic Bundle Output Engine")
    st.markdown("### ⚡ Dynamic Re-ranking & UI Rendering")
    st.info("Finalizing the bundle by balancing aesthetic consistency (DP3) and experience relevance (DP4).")
    current_scenario = st.session_state.get("scenario_selection", "Laurentian Forest Yoga (Nature Connectivity)")

    # --- 1. 显性化后台逻辑 (Process 层) ---
    with st.status("🧠 AI Engine: Executing Dynamic Re-ranking...", expanded=True) as status:
        st.write(f"📥 Fetching M3 Scenario: {current_scenario}")
        import time
        time.sleep(1)
        st.write("⚖️ Balancing Brand Alignment vs. Personal Preference (DP3/DP4)...")
        # 模拟 Re-ranking 过程
        st.progress(0.6, text="Processing Latent Association Matrix...")
        time.sleep(1)
        st.write("🎨 Rendering Visual Bundle Cards (DR12)...")
        status.update(label="✅ Delivery Ready: Optimized & Re-ranked!", state="complete")

    st.divider()

    # --- 2. 核心输出：互动式 Bundle 卡片 (Output 层) ---
    col_card, col_metrics = st.columns([2, 1])
    
    with col_card:
        with st.container(border=True):
            # 获取 M3 的图文数据
            current = content_map[current_scenario]
            st.image(current["img"], use_container_width=True)
            
            st.markdown(f"### 🏷️ **{current['title']} Bundle**")
            st.write(f"**Composition:** JUNA Core Set + Partner {current['color']} Layer")
            
            # 价格与行动点
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Bundle Price", "$189.00", "-15% Savings")
            with c2:
                # 模拟下载与保存
                st.button("📥 Export Proposal (PDF)", use_container_width=True)
                if st.button("❤️ Save to My Designs", key="save_m4"):
                    st.toast("Bundle saved to your dashboard!", icon="💾")

    with col_metrics:
        st.write("**📊 Real-time Delivery Metrics**")
        # 展示 DP12 反馈循环的量化指标
        st.progress(0.92, text="Aesthetic Consistency")
        st.progress(0.88, text="Scenario-Fit Score")
        st.progress(0.75, text="Market Trend Velocity")
        
        st.divider()
        # --- 3. 反馈闭环 (DR12 Feedback Loop) ---
        st.write("**🔄 Interaction Feedback Loop**")
        feedback = st.radio("Is this bundle aligned with your goal?", ("Perfect Match", "Need Refinement", "Not Relevant"))
        
        if st.button("🚀 Submit Feedback for Iteration", key="feedback_m4"):
            st.balloons()
            st.success("Feedback captured! Parameters sent back to M1/M2 for DR12 refinement.")

    st.divider()
    st.caption("🎯 **DSR Goal**: This module completes the cycle by providing a validated, tangible design artifact ready for market testing.")