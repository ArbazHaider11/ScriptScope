import streamlit as st
import os
from ingest import process_pdf_to_pinecone
from engine import get_analysis_chain

st.set_page_config(page_title="ScriptScope AI", page_icon="🎬")
st.title("🎬 ScriptScope: AI Movie Script Doctor")
st.caption("Developed by **Arbaz Haider** | Powered by Gemini 2.5 & Pinecone")

with st.sidebar:
    st.header("🔑 API Keys")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    google_api_key = st.text_input("Google API Key", type="password")
    index_name = st.text_input("Index Name", value="scriptscope")
    
    st.divider()

# --- MAIN PAGE ---
uploaded_file = st.file_uploader("Upload Movie Script (PDF)", type="pdf")

if uploaded_file and pinecone_api_key and google_api_key:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Indexing Button
    if st.button("Index Script"):
        with st.spinner("Embedding script into Pinecone..."):
            try:
                # Fix for Ingest ENV issue
                os.environ["PINECONE_API_KEY"] = pinecone_api_key
                
                process_pdf_to_pinecone(
                    pdf_path="temp.pdf",
                    api_key=pinecone_api_key,
                    index_name=index_name
                )
                st.success("Script indexed successfully!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    # Chat Interface
    user_query = st.text_input("Ask the Script Doctor:")

    if user_query:
        with st.spinner("Analyzing with Gemini..."):
            try:
                chain = get_analysis_chain(
                    pinecone_api_key=pinecone_api_key,
                    index_name=index_name,
                    google_api_key=google_api_key
                )
                report = chain.invoke(user_query)

                st.markdown("### 📝 Analysis Report")
                st.write(report)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

elif uploaded_file and (not pinecone_api_key or not google_api_key):
    st.warning("⚠️ Please enter both Pinecone and Google API keys in the sidebar.")