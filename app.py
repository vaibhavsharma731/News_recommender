"""
Modern Editorial News Platform & AI Recommendation Portal
"""
import streamlit as st
import pandas as pd
import config
from src.data_loader import load_articles
from src.recommender import METHODS, Recommender

st.set_page_config(
    page_title="Indian Express AI | News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------ CSS Injection for Premium UI
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F141C 0%, #1E293B 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border-left: 5px solid #E50914;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94A3B8;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .badge-live {
        background-color: #EF4444;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-section {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .badge-score {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }

    .article-card {
        background-color: #1A212D;
        border: 1px solid #2E3748;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .article-card:hover {
        border-color: #38BDF8;
    }
    
    .hero-card {
        background: linear-gradient(180deg, #1A212D 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ Recommender Initialization
@st.cache_resource(show_spinner=False)
def get_recommender() -> Recommender:
    with st.status("Starting AI News Engine...", expanded=True) as s:
        st.write("📂 Loading and cleaning 699 articles...")
        df = load_articles(config.DATA_PATH)
        st.write("🧠 Loading embedding model (bge-base-en-v1.5)...")
        rec = Recommender(df)
        st.write("✅ All systems ready!")
        s.update(label="News engine ready!", state="complete", expanded=False)
    return rec


recommender = get_recommender()
articles = recommender.articles

# Session State for User Reading History
if "user_history" not in st.session_state:
    st.session_state["user_history"] = []

# Reset card render counter each page run (guarantees unique button keys)
st.session_state["_card_n"] = 0


# ------------------------------------------------------------------ Sidebar Controls & User Drawer
with st.sidebar:
    st.image("https://images.indianexpress.com/2026/09/Article-Image_2.jpg?resize=450,253", use_container_width=True)

    st.title("⚙️ RecSys Controls")
    
    top_k = st.slider("Recommendations Count", 3, 15, config.TOP_K)
    section_filter = st.selectbox("Category Filter", ["All"] + sorted(articles["section"].unique()))
    
    st.divider()
    
    st.subheader("👤 Your Reading History")
    history_ids = st.session_state["user_history"]
    if history_ids:
        st.caption(f"{len(history_ids)} articles read in this session")
        for hid in reversed(history_ids[-5:]):
            art = recommender.get_article(hid)
            st.markdown(f"• **{art['title'][:45]}...**")
        if st.button("🗑️ Clear History"):
            st.session_state["user_history"] = []
            st.rerun()
    else:
        st.info("Click 'Read Story' on articles to personalize your 'For You' feed.")

    st.divider()
    
    with st.expander("🛠️ Advanced Engine Specs"):
        method = st.selectbox("Retrieval Method", METHODS, index=len(METHODS) - 1)
        use_boosts = st.checkbox("Calibrated Recency & Section Boosts", value=True)
        diversify = st.checkbox("MMR Diversity", value=True)
        show_images = st.checkbox("Show Images", value=True)
        
    st.caption(f"**Engine**: {recommender.engine_summary}")
    st.caption(f"**Indexed Collection**: {len(articles)} clean articles")
    for note in recommender.notes:
        st.warning(note)


# ------------------------------------------------------------------ Header Banner
st.markdown("""
<div class="main-header">
    <h1>THE INDIAN EXPRESS</h1>
    <p>AI-Powered Personalization & Intelligent News Discovery Engine</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ UI Helpers
def add_to_history(article_id):
    if article_id not in st.session_state["user_history"]:
        st.session_state["user_history"].append(article_id)
        st.toast("Added to reading history. Your 'For You' feed is updating!", icon="✅")


def show_article_card(article: dict, score: float = None, reason: str = None, images: bool = True, is_hero: bool = False):
    st.session_state["_card_n"] += 1
    card_n = st.session_state["_card_n"]
    card_class = "hero-card" if is_hero else "article-card"
    with st.container():
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        has_image = images and article["image"]
        cols = st.columns([1.2, 3]) if has_image else [st.container()]
        
        if has_image:
            try:
                cols[0].image(article["image"], use_container_width=True)
            except Exception:
                pass
            text_col = cols[1]
        else:
            text_col = cols[0]
            
        with text_col:
            # Badges
            badge_html = f'<span class="badge-section">{article["section"].upper()}</span> '
            if article["subsection"]:
                badge_html += f'<span class="badge-section">{article["subsection"].upper()}</span> '
            if article["is_live"]:
                badge_html += '<span class="badge-live">🔴 LIVE</span> '
            if score is not None:
                badge_html += f'<span class="badge-score">Match: {score:.0%}</span>'
            
            st.markdown(badge_html, unsafe_allow_html=True)
            
            # Title & Link
            title_link = f"[{article['title']}]({article['url']})" if article["url"] else article["title"]
            st.markdown(f"### {title_link}" if is_hero else f"#### {title_link}")
            
            st.caption(f"📅 Published: {article['date']}" if article['date'] else "")
            st.write(article["snippet"])
            
            if reason:
                st.info(f"💡 **Why recommended**: {reason}")
                
            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                if st.button("📖 Read Story", key=f"read_{article['id']}_{card_n}"):
                    add_to_history(article["id"])
                    st.rerun()
            with btn_col2:
                if article["url"]:
                    st.markdown(f"[🔗 Full Article on IndianExpress]({article['url']})")  
        st.markdown('</div>', unsafe_allow_html=True)


def display_results(result):
    for note in result.notes:
        if "Auto-corrected" in note:
            st.success(f"🔍 **Smart Indian Express Search:** Resolved query to {note.replace('Auto-corrected query terms: ', '')}")
        else:
            st.info(note)
    if not result.recommendations:
        st.warning("No relevant articles match your criteria. Try different words or remove category filters.")
        return
    if result.low_confidence:
        st.warning("⚡ No exact match found in current edition dataset. Showing nearest contextual matches.")
        
    st.subheader(f"Recommended News Articles ({len(result.recommendations)})")
    for rec in result.recommendations:
        show_article_card(rec.article, rec.score, rec.reason, show_images)


# ------------------------------------------------------------------ Featured Hero News Story
top_story = articles.sort_values("date", ascending=False).iloc[0]
with st.expander("🔥 TODAY'S SPOTLIGHT BREAKING STORY", expanded=False):
    show_article_card(recommender.get_article(top_story["id"]), is_hero=True, images=show_images)


# ------------------------------------------------------------------ Navigation Tabs
tab_foryou, tab_article, tab_search = st.tabs([
    "🎯 For You (Personalized Feed)",
    "📄 Item-to-Item Recommendations",
    "🔎 AI Semantic Search"
])

options = dict(
    top_k=top_k,
    method=method,
    use_boosts=use_boosts,
    diversify=diversify,
    section=None if section_filter == "All" else section_filter
)

# 1. Personalized "For You" Feed Tab
with tab_foryou:
    st.subheader("🎯 Personalized For You")
    if st.session_state["user_history"]:
        st.markdown("Your feed is dynamically personalized based on your active session reading profile.")
        res = recommender.recommend(user_history=st.session_state["user_history"], **options)
        display_results(res)
    else:
        st.info("💡 **Welcome!** Read 1 or 2 articles across any tab (click **'📖 Read Story'**) to activate your personalized AI feed!")
        # Fallback to recent trending stories
        st.markdown("##### Trending Top Stories Across Categories")
        recent_ids = articles.sort_values("date", ascending=False).head(top_k)["id"].tolist()
        for rid in recent_ids:
            show_article_card(recommender.get_article(rid), images=show_images)

# 2. Item-to-Item Similar Stories Tab
with tab_article:
    st.subheader("📄 Find Articles Similar to a Selected Story")
    sorted_arts = articles.sort_values("date", ascending=False)
    title_map = dict(zip(sorted_arts["id"], sorted_arts["title"] + "  (" + sorted_arts["section"] + ")"))
    
    selected_id = st.selectbox("Select or type to search an article:", list(title_map), format_func=title_map.get)
    st.markdown("##### Selected Base Article:")
    show_article_card(recommender.get_article(selected_id), images=show_images)
    
    st.divider()
    st.markdown("##### Related Recommended Articles:")
    res_item = recommender.recommend(article_id=selected_id, user_history=st.session_state["user_history"], **options)
    display_results(res_item)

# 3. AI Semantic Search Tab
with tab_search:
    st.subheader("🔎 Search Articles by Concept, Event, or Keywords")
    
    # Quick search shortcuts for testing
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    quick_query = None
    if col_p1.button("🗳️ Akhilesh Yadav (UP)", use_container_width=True):
        quick_query = "Akhilesh Yadav UP politics"
    if col_p2.button("🏏 Virat Kohli (Cricket)", use_container_width=True):
        quick_query = "Virat Kohli cricket centuries"
    if col_p3.button("⚖️ Supreme Court (SIR/EC)", use_container_width=True):
        quick_query = "Supreme Court Election Commission"
    if col_p4.button("🥇 Asian Games Medals", use_container_width=True):
        quick_query = "Asian Games gold medals India"

    user_query = st.text_input(
        "Enter any headline, politician name, event, or typo (e.g. 'akkhhlesh', 'virat koli'):",
        value=quick_query if quick_query else "",
        placeholder="e.g. akkhhlesh | Akhilesh Yadav UP elections | virat koli | Supreme Court ECI"
    )
    
    if user_query or st.button("Search Articles", type="primary", use_container_width=False):
        if user_query:
            res_query = recommender.recommend(query=user_query, user_history=st.session_state["user_history"], **options)
            display_results(res_query)
