import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import importlib

# Ensure custom backend utils reload cleanly
import utils
importlib.reload(utils)
from sample_article import SAMPLE_ARTICLE_TEXT

# ==========================================
# 1. PAGE CONFIGURATION & DARK THEME
# ==========================================
st.set_page_config(
    page_title="News AI Dashboard",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# High-end Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19; 
        color: #E2E8F0; 
    }
    .main-terminal-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        color: #00F2FE; 
        border-bottom: 2px solid #1E293B;
        padding-bottom: 12px;
        margin-bottom: 25px;
    }
    .section-banner {
        font-family: 'Inter', sans-serif;
        color: #38BDF8;
        font-weight: 600;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
        margin-top: 15px;
    }
    div[data-testid="stVerticalBlockBorderDiv"] {
        background: #111827 !important; 
        border: 1px solid #1F2937 !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }
    blockquote {
        border-left: 4px solid #10B981 !important; 
        background-color: #1F2937;
        color: #D1D5DB;
        padding: 15px;
        border-radius: 4px;
    }
    .stButton>button {
        background-color: #1F2937;
        color: #F3F4F6;
        border: 1px solid #4B5563;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #00F2FE;
        color: #00F2FE;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. COMPLETE SESSION STATE KEYS
# ==========================================
state_keys = [
    "article_text", "url", "nlp_data", "summary", "llm_sentiment", 
    "entities", "quotes", "chat_history", "bias_analysis", 
    "translated_content", "selected_lang", "jargon_glossary", 
    "briefing_script", "style_transformer", "factcheck_queries", 
    "compare_article_text", "compare_url", "compare_analysis", 
    "cultural_neutralizer", "echo_chamber_rewrite", "selected_echo_style"
]

for key in state_keys:
    if key not in st.session_state:
        if key in ["chat_history", "factcheck_queries"]:
            st.session_state[key] = []
        elif key == "nlp_data":
            st.session_state[key] = None
        else:
            st.session_state[key] = ""

# ==========================================
# 3. CONTROL PANEL & STATUS
# ==========================================
api_key_check = os.environ.get("NARAROUTER_API_KEY", "")
is_system_ready = bool(api_key_check)

with st.sidebar:
    st.markdown("### 🧭 Main Menu")
    
    workspace_mode = st.radio(
        "Choose a feature:",
        [
            "📥 1. Load an Article",
            "📝 2. Summary & Sentiment",
            "🔍 3. Key People & Details",
            "⚖️ 4. Check for Bias & Spin",
            "🤝 5. Compare Two Articles",
            "💬 6. Chat with the Article"
        ]
    )
    
    st.markdown("---")
    st.markdown("### AI Connection Status")
    if is_system_ready:
        st.success("🟢 AI Connected & Ready")
    else:
        st.error("🔴 API Key Missing")
        
    if st.button("🗑️ Clear Everything & Start Over", use_container_width=True):
        for key in state_keys:
            if key in ["chat_history", "factcheck_queries"]: st.session_state[key] = []
            elif key == "nlp_data": st.session_state[key] = None
            else: st.session_state[key] = ""
        st.rerun()

st.markdown('<div class="main-terminal-title">AI News Analyzer</div>', unsafe_allow_html=True)

# ==========================================
# 4. MONITOR LAYER CONTROLLERS
# ==========================================

# --- LAYER 1: DATA INGESTION HUB ---
if workspace_mode == "📥 1. Load an Article":
    st.markdown('<h3 class="section-banner">Add a News Article to Analyze</h3>', unsafe_allow_html=True)
    
    c_left, c_right = st.columns([2, 1])
    with c_left:
        with st.container(border=True):
            input_type = st.selectbox("How do you want to add the article?", ["Paste a Web Link (URL)", "Paste the Text Manually", "Use a Demo Article"])
            
            if input_type == "Paste a Web Link (URL)":
                target_url = st.text_input("Enter the article link:")
                if st.button("Load Article from Link", type="primary"):
                    if target_url:
                        with st.spinner("Downloading the article text..."):
                            st.session_state.article_text = utils.fetch_article_text(target_url)
                            st.session_state.url = target_url
                            st.success("Article loaded successfully!")
                            st.rerun()
                            
            elif input_type == "Paste the Text Manually":
                pasted_content = st.text_area("Paste the article text here:", height=200)
                if st.button("Save Pasted Text", type="primary"):
                    if len(pasted_content) > 50:
                        st.session_state.article_text = pasted_content
                        st.session_state.url = "Manually Pasted Text"
                        st.success("Article saved successfully!")
                        st.rerun()
                        
            else:
                if st.button("Load the NASA Demo Article", type="primary"):
                    st.session_state.article_text = SAMPLE_ARTICLE_TEXT
                    st.session_state.url = "NASA Deep Space Discovery (Demo)"
                    st.rerun()
                    
    with c_right:
        with st.container(border=True):
            st.markdown("#### Currently Loaded Article")
            if st.session_state.article_text:
                st.info(f"**Source:** {st.session_state.url[:35]}...")
                st.metric("Total Words", len(st.session_state.article_text.split()))
            else:
                st.warning("No article loaded yet. Please add one on the left.")

# --- LAYER 2: SUMMARY & SENTIMENT ---
elif workspace_mode == "📝 2. Summary & Sentiment":
    if not st.session_state.article_text:
        st.warning("Please load an article in Tab 1 first.")
    else:
        st.markdown('<h3 class="section-banner">Quick Summary & Emotional Tone</h3>', unsafe_allow_html=True)
        
        c_s, c_p = st.columns([1, 1])
        with c_s:
            with st.container(border=True):
                st.markdown("#### AI Bullet Point Summary")
                if not st.session_state.summary:
                    if st.button("Generate Summary"):
                        with st.spinner("Writing the summary..."):
                            st.session_state.summary = utils.query_nararouter(
                                f"Provide a simple, professional summary of this text in 4 distinct bullet points.\n\n{st.session_state.article_text}",
                                "You are a helpful news summarizer."
                            )
                        st.rerun()
                else:
                    st.write(st.session_state.summary)
                    
        with c_p:
            with st.container(border=True):
                st.markdown("#### Sentiment Analysis (Positive or Negative?)")
                vader_scores = utils.get_vader_sentiment(st.session_state.article_text)
                st.markdown(f"**Basic Score:** `{vader_scores['label']}` (Value: `{vader_scores['scores']['compound']:.2f}`)")
                
                if not st.session_state.llm_sentiment:
                    if st.button("Ask AI for Detailed Sentiment"):
                        with st.spinner("Analyzing the author's tone..."):
                            st.session_state.llm_sentiment = utils.query_nararouter(
                                f"Tell me if the overall sentiment of this article is POSITIVE, NEGATIVE, or NEUTRAL. Explain why based on the words the author used.\n\n{st.session_state.article_text}",
                                "You are an expert at understanding emotional tone in writing."
                            )
                        st.rerun()
                else:
                    st.markdown("---")
                    st.write(st.session_state.llm_sentiment)

# --- LAYER 3: KEY PEOPLE & DETAILS ---
elif workspace_mode == "🔍 3. Key People & Details":
    if not st.session_state.article_text:
        st.warning("Please load an article in Tab 1 first.")
    else:
        st.markdown('<h3 class="section-banner">Extract Important Details and Hard Words</h3>', unsafe_allow_html=True)
        
        c_e, c_t = st.columns([1, 1])
        with c_e:
            with st.container(border=True):
                st.markdown("#### Important People, Places, & Quotes")
                if not st.session_state.entities:
                    if st.button("Extract Key Information"):
                        with st.spinner("Finding people, companies, and quotes..."):
                            st.session_state.entities = utils.query_nararouter(
                                f"List the main People, Companies, and Places mentioned in this text. Use a simple bulleted list.\n\n{st.session_state.article_text}",
                                "You are a helpful assistant."
                            )
                            st.session_state.quotes = utils.query_nararouter(
                                f"Find the 3 most important direct quotes from this text. Put them in blockquotes.\n\n{st.session_state.article_text}",
                                "You are a helpful assistant."
                            )
                        st.rerun()
                else:
                    st.write(st.session_state.entities)
                    st.markdown("---")
                    st.write(st.session_state.quotes)
                    
        with c_t:
            with st.container(border=True):
                st.markdown("#### Jargon Dictionary")
                if not st.session_state.jargon_glossary:
                    if st.button("Explain Hard Words"):
                        with st.spinner("Creating a dictionary for this article..."):
                            st.session_state.jargon_glossary = utils.query_nararouter(
                                f"Find any complex words, technical jargon, or acronyms in this text and provide simple, plain-English definitions for them.\n\n{st.session_state.article_text}",
                                "You are a helpful teacher explaining hard words simply."
                            )
                        st.rerun()
                else:
                    st.write(st.session_state.jargon_glossary)

# --- LAYER 4: BIAS & SPIN CHECK ---
elif workspace_mode == "⚖️ 4. Check for Bias & Spin":
    if not st.session_state.article_text:
        st.warning("Please load an article in Tab 1 first.")
    else:
        st.markdown('<h3 class="section-banner">Analyze How the News is Framed</h3>', unsafe_allow_html=True)
        
        tab_audit, tab_neutral, tab_shift = st.tabs(["📊 Bias Report", "🕊️ Make it Neutral", "🔄 Rewrite the Style"])
        
        with tab_audit:
            with st.container(border=True):
                if not st.session_state.bias_analysis:
                    if st.button("Check for Media Bias"):
                        with st.spinner("Looking for emotional words and spin..."):
                            st.session_state.bias_analysis = utils.query_nararouter(
                                f"Check this text for media bias. Point out any emotionally loaded words, spin, or clickbait, and rate how sensational it is out of 10.\n\n{st.session_state.article_text}",
                                "You are a strict media critic looking for bias."
                            )
                        st.rerun()
                else:
                    st.write(st.session_state.bias_analysis)
                    
        with tab_neutral:
            with st.container(border=True):
                st.markdown("#### Neutral Version")
                if not st.session_state.cultural_neutralizer:
                    if st.button("Rewrite without Emotion or Spin"):
                        with st.spinner("Removing bias and rewriting facts..."):
                            st.session_state.cultural_neutralizer = utils.query_nararouter(
                                f"Rewrite this text to be 100% boring, factual, and neutral. Remove all opinions, spin, and emotional words.\n\n{st.session_state.article_text}",
                                "You are a totally neutral, factual robot."
                            )
                        st.rerun()
                else:
                    st.info(st.session_state.cultural_neutralizer)
                    
        with tab_shift:
            with st.container(border=True):
                st.markdown("#### Rewrite in a Different Style")
                style_var = st.selectbox("Choose a writing style:", ["Crazy Tabloid Gossip", "Boring Scientific Paper", "Tech Startup Pitch"])
                if st.button("Rewrite Article"):
                    with st.spinner(f"Rewriting in the style of: {style_var}..."):
                        st.session_state.style_transformer = utils.query_nararouter(
                            f"Rewrite this story completely as a {style_var}.\n\nStory:\n{st.session_state.article_text}",
                            "You are a creative writer."
                        )
                    st.rerun()
                if st.session_state.style_transformer:
                    st.markdown("---")
                    st.write(st.session_state.style_transformer)

# --- LAYER 5: COMPARE TWO ARTICLES ---
elif workspace_mode == "🤝 5. Compare Two Articles":
    if not st.session_state.article_text:
        st.warning("Please load your first article in Tab 1 before comparing.")
    else:
        st.markdown('<h3 class="section-banner">See How Different Sites Cover the Same Story</h3>', unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            with st.container(border=True):
                st.markdown("#### Article 1 (Already Loaded)")
                st.info(f"**Source:** {st.session_state.url}")
                st.metric("Length", f"{len(st.session_state.article_text.split())} words")
                
        with col_c2:
            with st.container(border=True):
                st.markdown("#### Article 2 (New Source)")
                comp_url = st.text_input("Enter link for the second article:")
                if st.button("Load Second Article"):
                    if comp_url:
                        with st.spinner("Downloading second article..."):
                            st.session_state.compare_article_text = utils.fetch_article_text(comp_url)
                            st.session_state.compare_url = comp_url
                            st.success("Second article loaded.")
                            st.rerun()
                            
        if st.session_state.compare_article_text:
            st.markdown("---")
            if st.button("Compare Both Articles", type="primary", use_container_width=True):
                with st.spinner("Finding differences in facts and tone..."):
                    st.session_state.compare_analysis = utils.query_nararouter(
                        f"Compare these two articles. Tell me what facts are missing from one, how their tone differs, and how they spin the story differently.\n\nArticle 1:\n{st.session_state.article_text}\n\nArticle 2:\n{st.session_state.compare_article_text}",
                        "You are an expert at comparing news articles."
                    )
                st.rerun()
                
            if st.session_state.compare_analysis:
                with st.container(border=True):
                    st.markdown("#### Comparison Results")
                    st.write(st.session_state.compare_analysis)

# --- LAYER 6: CHAT WITH THE ARTICLE ---
elif workspace_mode == "💬 6. Chat with the Article":
    if not st.session_state.article_text:
        st.warning("Please load an article in Tab 1 first.")
    else:
        st.markdown('<h3 class="section-banner">Ask Questions About the Text</h3>', unsafe_allow_html=True)
        st.markdown("The AI will only answer using facts from the article. It will not guess or make things up.")
        
        with st.container(border=True):
            for role, string in st.session_state.chat_history:
                if role == "User":
                    st.markdown(f"**👤 You:** {string}")
                else:
                    st.info(f"**🤖 AI:** {string}")
                    st.markdown("---")
                    
        user_prompt = st.text_input("Type your question here:", placeholder="e.g., Who is quoted in this article?")
        if st.button("Ask AI"):
            if user_prompt:
                st.session_state.chat_history.append(("User", user_prompt))
                node_response = utils.query_nararouter(
                    f"Answer this question using ONLY the facts explicitly provided in the text. If the answer is not in the text, say 'I cannot find the answer in this article.'\n\nArticle Text:\n{st.session_state.article_text}\n\nQuestion:\n{user_prompt}",
                    "You are a helpful assistant."
                )
                st.session_state.chat_history.append(("System", node_response))
                st.rerun()