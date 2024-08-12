import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import streamlit as st

# 간단한 사용자 데이터베이스 역할을 할 딕셔너리
if "user_db" not in st.session_state:
    st.session_state["user_db"] = {}

# 로그인 상태 확인 및 설정 함수
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

# 로그인 페이지
def login_page():
    st.title("로그인")
    id_input = st.text_input("아이디")
    pw_input = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        if id_input in st.session_state["user_db"] and st.session_state["user_db"][id_input] == pw_input:
            st.session_state["logged_in"] = True
            st.success("로그인 성공!")
        else:
            st.error("아이디 또는 비밀번호가 잘못되었습니다.")

    if st.button("회원가입"):
        st.session_state["current_page"] = "회원가입"

    if st.button("아이디/비밀번호 찾기"):
        st.session_state["current_page"] = "아이디/비밀번호 찾기"

# 회원가입 페이지
def signup_page():
    st.title("회원가입")
    new_id = st.text_input("아이디")
    new_pw = st.text_input("비밀번호", type="password")
    confirm_pw = st.text_input("비밀번호 확인", type="password")

    if st.button("회원가입"):
        if new_id in st.session_state["user_db"]:
            st.error("이미 존재하는 아이디입니다.")
        elif new_pw != confirm_pw:
            st.error("비밀번호가 일치하지 않습니다.")
        else:
            st.session_state["user_db"][new_id] = new_pw
            st.success("회원가입이 완료되었습니다!")

    if st.button("로그인 페이지로 이동"):
        st.session_state["current_page"] = "로그인"  # 사용자가 클릭 시 로그인 페이지로 이동

# 아이디/비밀번호 찾기 페이지 (간단한 형태로 구현)
def recover_account_page():
    st.title("아이디/비밀번호 찾기")
    st.write("이 페이지는 아직 구현되지 않았습니다.")

    if st.button("로그인 페이지로 돌아가기"):
        st.session_state["current_page"] = "로그인"

# 챗봇 페이지
def chat_page():
    st.title("PPAP 챗봇")
    st.write("챗봇과 대화하는 페이지입니다.")

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
            query_embedding = self.embeddings.embed_query(query)
            doc_texts = [doc.page_content for doc in docs]

            doc_embeddings = [self.embeddings.embed_query(doc) for doc in doc_texts]
            scores = [torch.cosine_similarity(torch.tensor(query_embedding), 
                                              torch.tensor(doc_embedding), dim=0).item() 
                                              for doc_embedding in doc_embeddings]
            sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
            return sorted_docs
        
        def set_url_retriever(self, url_path: str):
            loader = WebBaseLoader(url_path).load()
            pages = self.text_splitter.split_documents(loader)

            vectorstore_faiss = FAISS.from_documents(pages, self.embeddings)
            bm25_retriever = BM25Retriever.from_documents(pages)
            faiss_retriever = vectorstore_faiss.as_retriever(search_kwargs={'k': 5})

            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )

            template = '''Answer the question based only on the following context:
            {context}

            Question: {question}
            '''        
            self.prompt = ChatPromptTemplate.from_template(template)
            self.rag_chain = (
                self.prompt | self.model | StrOutputParser()
            )        

        def set_pdf_retriever(self, pdf_file_path: str):
            loaders = PyPDFLoader(file_path=pdf_file_path).load()
            docs = self.text_splitter.split_documents(loaders)

            vectorstore_faiss = FAISS.from_documents(docs, self.embeddings)
            bm25_retriever = BM25Retriever.from_documents(docs)
            faiss_retriever = vectorstore_faiss.as_retriever(search_kwargs={'k': 5})

            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )

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
            reranked_docs = self.rerank(query, result)
            formatted_docs = self.format_docs(reranked_docs)
            response = self.rag_chain.invoke({"context": formatted_docs, "question": query})
            return response
                
        def clear(self):
            self.retriever = None
            self.rag_chain = None

    # Streamlit 앱 구성
    gpt_rag = RagEnsemble()
    pdf_file_path1 = 'C:\\Users\\rkdal\\작성지침.pdf'

    gpt_rag.set_pdf_retriever(pdf_file_path1)

    user_input = st.text_input("질문을 입력하세요: ")

    if st.button("질문하기"):
        if user_input:
            answer = gpt_rag.ask(user_input)
            st.write("답변: ", answer)
        else:
            st.write("질문을 입력하세요.")

# Main 함수에서 로그인 상태에 따라 페이지를 전환
def main():
    check_login()

    if not st.session_state["logged_in"]:
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "로그인"

        if st.session_state["current_page"] == "로그인":
            login_page()
        elif st.session_state["current_page"] == "회원가입":
            signup_page()
        elif st.session_state["current_page"] == "아이디/비밀번호 찾기":
            recover_account_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
