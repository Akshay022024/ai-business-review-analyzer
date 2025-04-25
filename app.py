import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# 🔐 API Key
api_key = os.getenv("OPENROUTER_API_KEY") # Replace with your OpenRouter API key

# 📋 Function to generate embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to create embeddings
def generate_embeddings(texts, model):
    return model.encode(texts)

# Function to save embeddings
def save_embeddings(embeddings, file_path="review_embeddings.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
    return file_path

# Function to load embeddings
def load_embeddings(file_path="review_embeddings.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

# RAG search function
def rag_search(query, df, embeddings, model, top_k=5):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Format context from top matches
    context = "\n\n".join([
        f"Review {i+1} (Similarity: {similarities[idx]:.2f}):\n{df.iloc[idx]['review_text']}" 
        for i, idx in enumerate(top_indices)
    ])
    
    return context, top_indices

# 📄 Page settings
st.set_page_config(
    page_title="Business Review Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# 🎨 Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# 🏢 Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.markdown("## Business Review Analyzer")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool helps businesses analyze customer reviews using AI:
    - Extract key themes and topics
    - Visualize sentiment trends
    - Identify customer complaints
    - Generate actionable insights
    """)
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Upload your Google Reviews CSV
    2. Run the AI analysis
    3. Explore the dashboard
    4. Search for specific insights
    """)
    st.markdown("---")
    st.markdown("Powered by Mixtral & Vector Embeddings")

# 📄 Main page
st.markdown('<div class="main-header">🧠 Business Review Analyzer</div>', unsafe_allow_html=True)
st.markdown("Transform customer feedback into actionable business intelligence")

# 📁 Upload
uploaded_file = st.file_uploader("Upload Google Reviews CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully.")
else:
    df = pd.read_csv("greenleaf_reviews.csv")
    st.info("📂 Using default data for demo.")

# ✅ Check required column
if "review_text" not in df.columns:
    st.error("❌ The CSV must contain a 'review_text' column.")
    st.stop()

# 🧠 Initialize embedding model
model = load_embedding_model()

# 📊 Tabs for organized workflow
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔍 RAG Search", "💡 AI Insights"])

with tab1:
    # 🌟 Review Analysis Button
    if st.button("🔍 Run AI Review Analysis", key="run_analysis"):
        with st.spinner("⏳ Analyzing reviews using Mixtral & generating embeddings..."):
            # Generate embeddings for all reviews
            embeddings = generate_embeddings(df["review_text"].tolist(), model)
            embedding_file = save_embeddings(embeddings)
            st.session_state['embeddings'] = embeddings
            st.session_state['embedding_file'] = embedding_file
            
            results = []

            col1, col2 = st.columns([1, 1])
            progress_bar = col1.progress(0)
            status_text = col2.empty()

            for i, row in df.iterrows():
                progress = (i + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing review {i+1} of {len(df)}")
                
                prompt = f"""
    Analyze this customer review:

    "{row['review_text']}"

    Format response strictly:
    Sentiment: Positive / Negative / Neutral  
    Complaint: Yes / No  
    Theme: (one/two keywords)  
    Polarity Score: (0 to 1)  
    Category: (e.g., staff, pricing, delivery)  
    Action Needed: (e.g., Train staff, Improve pricing)

    Answer:
    Sentiment: ...
    Complaint: ...
    Theme: ...
    Polarity Score: ...
    Category: ...
    Action Needed: ...
    """

                try:
                    res = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "mistralai/mixtral-8x7b-instruct",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 200,
                        },
                    )
                    reply = res.json()["choices"][0]["message"]["content"]

                    sentiment = complaint = theme = polarity = category = action = ""
                    for line in reply.splitlines():
                        if "Sentiment" in line: sentiment = line.split(":")[-1].strip()
                        elif "Complaint" in line: complaint = line.split(":")[-1].strip()
                        elif "Theme" in line: theme = line.split(":")[-1].strip()
                        elif "Polarity Score" in line: polarity = line.split(":")[-1].strip()
                        elif "Category" in line: category = line.split(":")[-1].strip()
                        elif "Action Needed" in line: action = line.split(":")[-1].strip()

                    results.append((sentiment, complaint, theme, polarity, category, action))

                except Exception as e:
                    results.append(("Error", "Error", "", "", "", ""))
                    st.warning(f"⚠️ Row {i+1} failed: {e}")

            # Add columns
            df[["Sentiment", "Complaint", "Theme", "Polarity Score", "Category", "Action Needed"]] = results
            
            # Store in session state
            st.session_state['analyzed_df'] = df
            
            # Convert Polarity Score to numeric
            df["Polarity Score"] = pd.to_numeric(df["Polarity Score"], errors="coerce")

            # Try parsing date if available
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                except Exception as e:
                    st.warning("📅 Date column found but could not be parsed. Skipping sentiment trend.")
                    df["date"] = None
            else:
                df["date"] = None
                
            st.success("✅ Analysis Complete.")

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Analyzed Data", csv, "analyzed_reviews.csv", "text/csv")

    # Display analysis results if available
    if 'analyzed_df' in st.session_state:
        df = st.session_state['analyzed_df']
        
        # 📊 DASHBOARD
        st.markdown('<div class="sub-header">📊 Dashboard Summary</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        total = len(df)
        pos_pct = round((df["Sentiment"].str.lower() == "positive").sum() / total * 100, 2)
        comp_pct = round((df["Complaint"].str.lower() == "yes").sum() / total * 100, 2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pos_pct}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">✅ Positive Reviews</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{comp_pct}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">⚠️ Complaints</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{total}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">📝 Total Reviews</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Visualization layout
        left_col, right_col = st.columns(2)
        
        # Sentiment Pie
        with left_col:
            st.markdown("### 🧠 Sentiment Breakdown")
            fig_pie = px.pie(
                df, 
                names="Sentiment", 
                title="Sentiment Distribution", 
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig_pie.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Category Bar
        with right_col:
            st.markdown("### 🏷️ Top Categories")
            cat_counts = df["Category"].value_counts().head(10)
            fig_bar = px.bar(
                cat_counts, 
                x=cat_counts.index, 
                y=cat_counts.values, 
                labels={"x": "Category", "y": "Count"}, 
                title="Top Categories"
            )
            fig_bar.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # Sentiment Trend (if date column exists)
        if "date" in df.columns and df["date"].notnull().all():
            st.markdown("### 📈 Sentiment Trend Over Time")
            df_trend = df.groupby(pd.Grouper(key="date", freq="W")).agg({"Polarity Score": "mean"}).reset_index()
            fig_trend = px.line(
                df_trend, 
                x="date", 
                y="Polarity Score", 
                title="Weekly Sentiment Trend", 
                markers=True
            )
            fig_trend.update_layout(
                yaxis_range=[0, 1], 
                xaxis_title="Date", 
                yaxis_title="Average Polarity"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Theme WordCloud Attempt
        st.markdown("### 🔤 Most Common Themes")
        theme_counts = df["Theme"].value_counts().head(15)
        fig_themes = px.bar(
            theme_counts, 
            x=theme_counts.index, 
            y=theme_counts.values, 
            labels={"x": "Themes", "y": "Count"}, 
            title="Common Themes in Reviews",
            color=theme_counts.values,
            color_continuous_scale="Viridis"
        )
        fig_themes.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_themes, use_container_width=True)

        # ❗ Priority Customers
        st.markdown("### 🚨 Priority Reviews")
        priority_df = df[(df["Sentiment"].str.lower() == "negative") | (df["Complaint"].str.lower() == "yes")]
        st.write("🧾 Customers to prioritize (based on sentiment or complaint):")
        st.dataframe(
            priority_df[["review_text", "Sentiment", "Complaint", "Category", "Action Needed"]],
            use_container_width=True,
            height=300
        )

        # Display full data table
        with st.expander("View All Analyzed Data"):
            st.dataframe(df, use_container_width=True, height=300)

with tab2:
    st.markdown('<div class="sub-header">🔍 RAG-powered Review Search</div>', unsafe_allow_html=True)
    st.markdown("""
    Search through your reviews using natural language queries. 
    The system will find the most relevant reviews based on semantic similarity.
    """)
    
    if 'embeddings' not in st.session_state:
        st.warning("⚠️ Please run the analysis first to generate embeddings.")
    else:
        user_query = st.text_input("What would you like to know about your reviews?", 
                                 placeholder="E.g., What are customers saying about our staff?")
        
        if user_query:
            with st.spinner("🔍 Searching through reviews..."):
                # Get context from RAG
                context, top_indices = rag_search(
                    user_query, 
                    st.session_state['analyzed_df'], 
                    st.session_state['embeddings'], 
                    model, 
                    top_k=5
                )
                
                # Format prompt with RAG context
                rag_prompt = f"""
You are a business insight assistant. A user asked: "{user_query}"

Use these most relevant customer reviews:
{context}

Provide a helpful, concise answer based specifically on these reviews. 
Include direct quotes where relevant. Be factual and specific.
"""
                
                try:
                    search_res = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "mistralai/mixtral-8x7b-instruct",
                            "messages": [{"role": "user", "content": rag_prompt}],
                            "max_tokens": 500,
                        },
                    )
                    search_reply = search_res.json()["choices"][0]["message"]["content"]
                    
                    # Display the answer
                    st.success("🔍 Here's what I found:")
                    st.markdown(search_reply)
                    
                    # Display the most relevant reviews
                    with st.expander("View most relevant reviews"):
                        for i, idx in enumerate(top_indices):
                            st.markdown(f"**Review {i+1}:**")
                            st.markdown(f"*{st.session_state['analyzed_df'].iloc[idx]['review_text']}*")
                            st.markdown("---")
                            
                except Exception as e:
                    st.error(f"❌ Search failed: {e}")

with tab3:
    st.markdown('<div class="sub-header">💡 AI Business Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    Get AI-generated business intelligence from your reviews.
    Choose a specific aspect to analyze:
    """)
    
    if 'analyzed_df' not in st.session_state:
        st.warning("⚠️ Please run the analysis first to generate insights.")
    else:
        df = st.session_state['analyzed_df']
        
        insight_type = st.selectbox(
            "What type of insight would you like?",
            [
                "Top business priorities",
                "Customer service assessment",
                "Product quality analysis",
                "Competitive advantages",
                "Growth opportunities",
                "Custom insight"
            ]
        )
        
        if insight_type == "Custom insight":
            custom_insight = st.text_input("What specific insight would you like?")
            if custom_insight:
                insight_prompt = f"""
You are a business analyst examining customer reviews. 

Based on the following review data:
{df[['review_text', 'Sentiment', 'Category', 'Theme', 'Action Needed']].head(50).to_string(index=False)}

The user wants insights about: {custom_insight}

Provide a detailed, data-driven analysis with specific examples from the reviews.
Include 3-5 actionable recommendations. Format your answer in markdown.
"""
        else:
            prompts = {
                "Top business priorities": "What are the top 3-5 urgent improvements the business should prioritize based on these reviews? Give clear, specific recommendations.",
                "Customer service assessment": "Analyze the customer service performance based on these reviews. What's working well and what needs improvement?",
                "Product quality analysis": "Analyze product quality mentions in these reviews. What do customers like and dislike about the products?",
                "Competitive advantages": "Based on positive comments in these reviews, what are this business's competitive advantages? What do customers particularly value?",
                "Growth opportunities": "What growth opportunities can be identified from these reviews? What new services or improvements could drive more business?"
            }
            
            insight_prompt = f"""
You are a business analyst examining customer reviews. 

Based on the following review data:
{df[['review_text', 'Sentiment', 'Category', 'Theme', 'Action Needed']].head(50).to_string(index=False)}

{prompts[insight_type]}

Provide a detailed, data-driven analysis with specific examples from the reviews.
Include 3-5 actionable recommendations. Format your answer in markdown.
"""
        
        if st.button("Generate Insight", key="generate_insight"):
            with st.spinner("🧠 Generating business intelligence..."):
                try:
                    insight_res = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "mistralai/mixtral-8x7b-instruct",
                            "messages": [{"role": "user", "content": insight_prompt}],
                            "max_tokens": 1000,
                        },
                    )
                    insight_reply = insight_res.json()["choices"][0]["message"]["content"]
                    
                    # Display the insight
                    st.markdown(insight_reply)
                    
                    # Option to download the insight
                    insight_text = f"# {insight_type} Analysis\n\n{insight_reply}"
                    st.download_button(
                        "📥 Download This Insight", 
                        insight_text.encode("utf-8"), 
                        f"{insight_type.lower().replace(' ', '_')}_insight.md", 
                        "text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Could not generate insight: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    Built with ❤️ using Streamlit, Mixtral, and Vector Embeddings
</div>
""", unsafe_allow_html=True)