"""Streamlit 진입점.

uv run streamlit run app.py --server.address localhost
"""

import streamlit as st

st.set_page_config(page_title="문서 파싱·추출", layout="wide")
st.navigation([
    st.Page("views/viewer.py", title="파싱·추출 확인", icon="📄", default=True),
    st.Page("views/playground.py", title="실험 플레이그라운드", icon="🧪", url_path="lab"),
]).run()
