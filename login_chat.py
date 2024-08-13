import os
from dotenv import load_dotenv
import torch
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

# API 키 정보 로드
load_dotenv()

# 간단한 사용자 데이터베이스 역할을 할 딕셔너리
if "user_db" not in st.session_state:
    st.session_state["user_db"] = {}

# 로그인 상태 확인 및 설정 함수
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

# 메인 페이지
def main_page():
    st.title("PPAP 서비스")

    ppap_login_html = '''
    <style>
        .ppap-login-btn {
            display: block;
            width: 250px;
            height: 50px;
            background-color: #4169E1;
            color: white;
            text-align: center;
            line-height: 50px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 16px;
            margin: 0 auto;
            cursor: pointer;
        }
    </style>
    <div class="ppap-login-btn" onclick="location.href='/로그인'">PPAP 로그인</div>
    '''
    st.markdown(ppap_login_html, unsafe_allow_html=True)

    # 아이디, 비밀번호 찾기, 회원가입 링크
    st.markdown('''
    <div style="text-align: center; margin-top: 20px;">
        <a href="#" onclick="location.href='/아이디_찾기'">아이디 찾기</a> |
        <a href="#" onclick="location.href='/비밀번호_찾기'">비밀번호 찾기</a> |
        <a href="#" onclick="location.href='/회원가입'">회원가입</a>
    </div>
    ''', unsafe_allow_html=True)

# 로그인 페이지
def ppap_login_page():
    st.title("PPAP 로그인")
    id_input = st.text_input("아이디")
    pw_input = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        if id_input in st.session_state["user_db"] and st.session_state["user_db"][id_input] == pw_input:
            st.session_state["logged_in"] = True
            st.session_state["current_page"] = "챗봇"  # 로그인 성공 후 챗봇 페이지로 이동
            st.success("로그인 성공!")
        else:
            st.error("아이디 또는 비밀번호가 잘못되었습니다.")

    if st.button("회원가입 페이지로 이동"):
        st.session_state["current_page"] = "회원가입"

    if st.button("아이디/비밀번호 찾기 페이지로 이동"):
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
        st.session_state["current_page"] = "로그인"

# 아이디/비밀번호 찾기 페이지 (간단한 형태로 구현)
def recover_account_page():
    st.title("아이디/비밀번호 찾기")
    st.write("이 페이지는 아직 구현되지 않았습니다.")

    if st.button("로그인 페이지로 돌아가기"):
        st.session_state["current_page"] = "로그인"

# 챗봇 페이지
def chat_page():
    st.title("PPAP")
    st.write("개인정보처리방침 평가 어시스턴트 챗봇")

    # RagEnsemble 클래스 정의 및 초기화
    class RagEnsemble:
        def __init__(self):
            self.model = ChatOpenAI(model="gpt-4-turbo", temperature=0)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
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
            # 웹페이지 로드
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

    # RagEnsemble 클래스 인스턴스화 및 사용
    gpt_rag = RagEnsemble()
    pdf_file_path1 = 'C:\\Users\\rkdal\\작성지침.pdf'
    #pdf_file_path2 = '/Users/rkdal/24년 개인정보 처리방침 평가계획.pdf'

    gpt_rag.set_pdf_retriever(pdf_file_path1)
    #gpt_rag.set_pdf_retriever(pdf_file_path2)

    # 챗봇 UI 구현
    question = st.text_input('질문 입력')
    submit = st.button('제출')
    if submit:
        answer = gpt_rag.ask(question)
        st.write('답변:', answer)

    if st.button('다른 PDF 파일로 설정'):
        gpt_rag.clear()

# 메인 실행 함수
def run_app():
    check_login()
    
    # current_page가 세션 상태에 없으면 메인 페이지로 설정
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "메인"

    # current_page에 따라 다른 페이지 호출
    if st.session_state["current_page"] == "메인":
        main_page()
    elif st.session_state["current_page"] == "로그인":
        ppap_login_page()
    elif st.session_state["current_page"] == "회원가입":
        signup_page()
    elif st.session_state["current_page"] == "아이디/비밀번호 찾기":
        recover_account_page()
    elif st.session_state["current_page"] == "챗봇":
        if st.session_state["logged_in"]:
            chat_page()
        else:
            st.warning("로그인이 필요합니다.")
            st.session_state["current_page"] = "로그인"

if __name__ == "__main__":
    run_app()
