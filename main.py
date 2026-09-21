import os
import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

# Get API key from .env or Streamlit Cloud secrets
google_api_key = os.environ.get("GOOGLE_API_KEY", "")
if not google_api_key:
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

st.title("RockyBot: News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=google_api_key,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

main_placeholder = st.empty()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


def scrape_article(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Try to extract main article content
    article = soup.find("article") or soup.find("main") or soup.find("div", class_=lambda c: c and any(x in c.lower() for x in ["article", "content", "story", "body"]))
    
    if article:
        text = article.get_text(separator="\n", strip=True)
    else:
        # Fallback: get all paragraphs
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)

    return text


if process_url_clicked:
    valid_urls = [u for u in urls if u.strip()]
    if not valid_urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        try:
            main_placeholder.text("Data Loading...Started...✅✅✅")
            docs = []
            for url in valid_urls:
                text = scrape_article(url)
                if text:
                    docs.append(Document(page_content=text, metadata={"source": url}))

            if not docs:
                main_placeholder.error("Could not extract content from the URLs. Try different articles.")
            else:
                main_placeholder.text(f"Loaded {len(docs)} articles. Splitting text...✅✅✅")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", ","],
                    chunk_size=1000
                )
                split_docs = text_splitter.split_documents(docs)

                main_placeholder.text(f"Building Embedding Vectors from {len(split_docs)} chunks...✅✅✅")
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                st.session_state.vectorstore = vectorstore
                time.sleep(1)
                main_placeholder.success(f"Done! Indexed {len(split_docs)} chunks from {len(docs)} articles. Ask your question below ✅")
        except Exception as e:
            main_placeholder.error(f"Error: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if st.session_state.vectorstore is None:
        st.warning("Please process URLs first using the sidebar.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(query)

        context = "\n\n".join(d.page_content for d in retrieved_docs)
        sources = list(set(
            d.metadata.get("source", "") for d in retrieved_docs if d.metadata.get("source")
        ))

        prompt = PromptTemplate.from_template("""You are a helpful news research assistant.
Answer the question based only on the provided context from news articles.
Be concise and specific. If the answer is not in the context, say "I don't know based on the provided articles."

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
                    st.error("Rate limit reached. Please wait a minute and try again.")
                else:
                    st.error(f"Error: {e}")

        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(source)
