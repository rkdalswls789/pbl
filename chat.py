import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser

from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate 
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import torch
import streamlit as st

# API 키 정보 로드
load_dotenv()

class RagEnsemble:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-m3',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True}
        )
        self.retriever = None
        self.rag_chain = None

    def format_docs(self, docs):
        return '\n\n'.join([d.page_content for d in docs])  

    def rerank(self, query, docs):
        query_embedding = self.embeddings.embed_query(query) # 질문 임베딩 생성
        doc_texts = [doc.page_content for doc in docs] # 문서 텍스트 추출

        doc_embeddings = [self.embeddings.embed_query(doc) for doc in doc_texts] # 문서 임베딩 생성
        scores = [torch.cosine_similarity(torch.tensor(query_embedding), 
                                          torch.tensor(doc_embedding), dim=0).item() 
                                          for doc_embedding in doc_embeddings] # 코사인 유사도 계산
        sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)] # 유사도 기준 정렬
        return sorted_docs
    
    def set_url_retriever(self, url_path: str):
        #웹페이지 로드
        loader = WebBaseLoader(url_path).load()
        pages = self.text_splitter.split_documents(loader)

        # FAISS 벡터 스토어 생성
        vectorstore_faiss = FAISS.from_documents(pages, self.embeddings)
        # BM25 리트리버 생성
        bm25_retriever = BM25Retriever.from_documents(pages)
        # FAISS 리트리버 생성
        faiss_retriever = vectorstore_faiss.as_retriever(search_kwargs={'k': 5})

        # Ensemble 리트리버 생성
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5] # 가중치 설정
        )

        # 답변 생성을 위한 프롬프트 템플릿 설정
        template = '''Answer the question based only on the following context:
        {context}

        Question: {question}
        '''        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.rag_chain = (
            self.prompt | self.model | StrOutputParser()
        )        

    def set_pdf_retriever(self, pdf_file_path: str):
        # PDF 파일 로드
        loaders = PyPDFLoader(file_path=pdf_file_path).load()
        # 문서 분할
        docs = self.text_splitter.split_documents(loaders)

        # FAISS 벡터 스토어 생성
        vectorstore_faiss = FAISS.from_documents(docs, self.embeddings)
        # BM25 리트리버 생성
        bm25_retriever = BM25Retriever.from_documents(docs)
        # FAISS 리트리버 생성
        faiss_retriever = vectorstore_faiss.as_retriever(search_kwargs={'k': 5})

        # Ensemble 리트리버 생성
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5] # 가중치 설정
        )

        # 답변 생성을 위한 프롬프트 템플릿 설정
        template = '''Answer the question based only on the following context:
        {context}

        Question: {question}
        '''        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.rag_chain = (
            self.prompt | self.model | StrOutputParser()
        )        
    
    def ask(self, query: str):
        if not self.rag_chain:
            return "평가할 PDF 파일을 먼저 등록해주세요"
        result = self.retriever.get_relevant_documents(query)
        reranked_docs = self.rerank(query, result) # 문서 재정렬
        formatted_docs = self.format_docs(reranked_docs) # 문서 포맷팅
        # 답변 생성
        response = self.rag_chain.invoke({"context": formatted_docs, "question": query})
        return response
            
    def clear(self):
        # 설정 초기화
        self.retriever = None
        self.rag_chain = None

# Streamlit 앱 구성
def main():
    st.title("PPAP")
    st.write("개인정보처리방침 평가 어시스턴트 챗봇")

    gpt_rag = RagEnsemble()
    pdf_file_path1 = 'C:\Users\rkdal\작성지침.pdf'
    #pdf_file_path2 = '/Users/rkdal/24년 개인정보 처리방침 평가계획.pdf'

    gpt_rag.set_pdf_retriever(pdf_file_path1)
    #gpt_rag.set_pdf_retriever(pdf_file_path2)

    user_input = st.text_input("질문을 입력하세요: ")

    if st.button("질문하기"):
        if user_input:
            answer = gpt_rag.ask(user_input)
            st.write("답변: ", answer)
        else:
            st.write("질문을 입력하세요.")

if __name__ == "__main__":
    main()
