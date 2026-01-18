import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load Embeddings (Keep this compatible with your ingest.py)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. The Analysis Chain using Gemini
def get_analysis_chain(pinecone_api_key: str, index_name: str, google_api_key: str):
    
    # Fix Pinecone ENV issue
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    embeddings = load_embeddings()

    # Connect to Pinecone
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Setup Gemini
    if not google_api_key:
        raise ValueError("Google API Key is missing.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_api_key
    )

    template = """
    You are a veteran Hollywood Script Doctor. Your client has sent you some pages to review.
    Do not answer out of context questions.
    Below are the relevant script segments they sent:
    "{context}"
    
    ---
    **Your Feedback:**
    Provide a direct, high-level critique. 
    1. **Don't fluff it up.** Do not say "Based on the provided text" or "As an AI". Jump straight into the answer.
    2. **Be specific.** Quote the exact lines that are working or failing.
    3. **Rewrite.** If you see a weak line (especially on-the-nose dialogue), show exactly how you would rewrite it to be subtextual.
    
    Write your response in clean Markdown.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain