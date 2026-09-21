import os
import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Get API key from .env or Streamlit Cloud secrets
google_api_key = os.environ.get("GOOGLE_API_KEY", "")
if not google_api_key:
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

main_placeholder = st.empty()

# Keep vectorstore in session state (works locally + Streamlit Cloud)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if process_url_clicked:
    valid_urls = [u for u in urls if u.strip()]
    if not valid_urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        try:
            main_placeholder.text("Data Loading...Started...✅✅✅")
            loader = WebBaseLoader(valid_urls)
            data = loader.load()

            main_placeholder.text("Text Splitting...Started...✅✅✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            main_placeholder.text("Building Embedding Vectors...✅✅✅")
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.vectorstore = vectorstore
            time.sleep(1)
            main_placeholder.success("Processing Complete! You can now ask questions ✅")
        except Exception as e:
            main_placeholder.error(f"Error processing URLs: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if st.session_state.vectorstore is None:
        st.warning("Please process URLs first using the sidebar.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        context = "\n\n".join(d.page_content for d in docs)
        sources = list(set(
            d.metadata.get("source", "") for d in docs if d.metadata.get("source")
        ))

        prompt = PromptTemplate.from_template("""You are a helpful news research assistant.
Answer the question based only on the provided context from news articles.
If the answer is not in the context, say "I don't know based on the provided articles."

Context:
{context}

Question: {question}

Answer:""")

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke({"context": context, "question": query})
                st.header("Answer")
                st.write(answer)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.error("⚠️ Rate limit reached. Please wait a minute and try again, or come back tomorrow for the free tier reset.")
                else:
                    st.error(f"Error: {e}")

        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(source)
