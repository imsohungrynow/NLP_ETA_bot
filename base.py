from functools import reduce
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

# 환경 변수를 로드하고 API 키를 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 언어 모델 초기화
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

# 텍스트 분할 설정
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=100,
)

# 문서 로드 및 분할
file_path = ""

def load_and_split(file_path):
    loader = CSVLoader(file_path, encoding='utf8')
    return loader.load_and_split(text_splitter=splitter)

docs = load_and_split(file_path)


# 임베딩과 캐싱 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 저장소 생성
vectorstore = FAISS.from_documents(docs, embeddings)

# 검색기 설정
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})