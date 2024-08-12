import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.vectorstores import FAISS, Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# 1. API 키 정보 로드
load_dotenv()

# 간단한 유저 데이터베이스 시뮬레이션 (실제 구현에서는 DB 사용 권장)
user_db = {}

# 2. RagEnsemble 클래스 정의
class RagEnsemble:
    def __init__(self):
        # 3. GPT-4 Turbo 모델 초기화
        self.model = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        
        # 4. 텍스트를 일정 크기로 분할할 수 있는 설정 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        # 5. HuggingFace 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-m3',
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings':True}
        )
        
        # 6. 리트리버와 RAG 체인을 위한 변수 초기화
        self.retriever = None
        self.rag_chain = None

    # 7. 문서들을 하나의 문자열로 포맷팅
    def format_docs(self, docs):
        return '\n\n'.join([d.page_content for d in docs])  

    # 8. 문서들을 재정렬하기 위한 함수 (코사인 유사도 기반)
    def rerank(self, query, docs):
        query_embedding = self.embeddings.embed_query(query) # 질문 임베딩 생성
        doc_texts = [doc.page_content for doc in docs] # 문서 텍스트 추출

        # 각 문서에 대한 임베딩 생성 및 유사도 계산
        doc_embeddings = [self.embeddings.embed_query(doc) for doc in doc_texts]
        scores = [torch.cosine_similarity(torch.tensor(query_embedding), 
                                          torch.tensor(doc_embedding), dim=0).item() 
                                          for doc_embedding in doc_embeddings]
        
        # 유사도 기준으로 문서 정렬
        sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return sorted_docs
    
    # 9. 웹페이지에서 문서를 로드하고 리트리버 설정
    def set_url_retriever(self, url_path: str):
        # 9-1. 웹페이지 로드 및 텍스트 분할
        loader = WebBaseLoader(url_path).load()
        pages = self.text_splitter.split_documents(loader)

        # 9-2. FAISS 벡터 스토어 생성 및 설정
        vectorstore_faiss = FAISS.from_documents(pages, self.embeddings)
        
        # 9-3. BM25 리트리버 생성 및 설정
        bm25_retriever = BM25Retriever.from_documents(pages)
        
        # 9-4. FAISS 리트리버 생성 및 설정
        faiss_retriever = vectorstore_faiss.as_retriever(search_kwargs={'k': 5})

        # 9-5. Ensemble 리트리버 생성 및 설정
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5] # 가중치 설정
        )

        # 9-6. 답변 생성을 위한 프롬프트 템플릿 설정
        template = '''Answer the question based only on the following context:
        {context}

        Question: {question}
        '''        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # 9-7. RAG 체인 설정
        self.rag_chain = (
            self.prompt | self.model | StrOutputParser()
        )        

    # 10. PDF에서 문서를 로드하고 리트리버 설정
    def set_pdf_retriever(self, pdf_file_path: str):
        # 10-1. PDF 파일 로드 및 텍스트 분할
        loaders = PyPDFLoader(file_path=pdf_file_path).load()
        docs = self.text_splitter.split_documents(loaders)

        # 10-2. FAISS 벡터 스토어 생성 및 설정
        vectorstore_faiss = FAISS.from_documents(docs, self.embeddings)
        
        # 10-3. BM25 리트리버 생성 및 설정
        bm25_retriever = BM25Retriever.from_documents(docs)
        
        # 10-4. FAISS 리트리버 생성 및 설정
        faiss_retriever = vectorstore_faiss.as_retriever(search_kwargs={'k': 5})

        # 10-5. Ensemble 리트리버 생성 및 설정
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5] # 가중치 설정
        )

        # 10-6. 답변 생성을 위한 프롬프트 템플릿 설정
        template = '''Answer the question based only on the following context:
        {context}

        Question: {question}
        '''        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # 10-7. RAG 체인 설정
        self.rag_chain = (
            self.prompt | self.model | StrOutputParser()
        )        
    
    # 11. 질문을 받아 처리하고 답변을 생성하는 함수
    def ask(self, query: str):
        if not self.rag_chain:
            return "평가할 PDF 파일을 먼저 등록해주세요"
        
        # 11-1. 질문에 대한 관련 문서 검색
        result = self.retriever.get_relevant_documents(query)
        
        # 11-2. 문서 재정렬
        reranked_docs = self.rerank(query, result)
        
        # 11-3. 문서를 포맷팅하여 답변 생성
        formatted_docs = self.format_docs(reranked_docs)
        response = self.rag_chain.invoke({"context": formatted_docs, "question": query})
        return response
            
    # 12. 설정 초기화
    def clear(self):
        self.retriever = None
        self.rag_chain = None

# 로그인 상태를 확인하는 함수
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

def login_page():
    st.title("PPAP 로그인")

    # 로그인 폼
    user_id = st.text_input("아이디")
    user_pw = st.text_input("비밀번호", type="password")
    if st.button("로그인"):
        if user_id in user_db and user_db[user_id] == user_pw:
            st.session_state['logged_in'] = True
            st.success("로그인 성공!")
        else:
            st.error("아이디 또는 비밀번호가 잘못되었습니다.")

    # ID/PW 찾기 및 회원가입 링크
    st.write("[회원가입](#회원가입)")
    st.write("[아이디/비밀번호 찾기](#)")

def signup_page():
    st.title("회원가입")

    # 회원가입 폼
    new_user_id = st.text_input("새 아이디")
    new_user_pw = st.text_input("새 비밀번호", type="password")
    if st.button("회원가입"):
        if new_user_id in user_db:
            st.error("이미 존재하는 아이디입니다.")
        else:
            user_db[new_user_id] = new_user_pw
            st.success("회원가입 성공! 로그인해주세요.")
            st.write("[로그인](#로그인)")

def chat_page():
    st.title("PPAP 챗봇")
    st.write("개인정보처리방침 평가 어시스턴트 챗봇")

    # 14. RagEnsemble 인스턴스 생성
    gpt_rag = RagEnsemble()
    
    # 15. PDF 파일 경로 설정 및 리트리버 초기화
    pdf_file_path1 = '/Users/kchabin/Desktop/PythonWorkSpace/rag/privacy/작성지침.pdf'
    gpt_rag.set_pdf_retriever(pdf_file_path1)

    # 16. 사용자 질문 입력 받기
    user_input = st.text_input("질문을 입력하세요: ")

    # 17. 버튼 클릭 시 질문 처리 및 답변 출력
    if st.button("질문하기"):
        if user_input:
            answer = gpt_rag.ask(user_input)
            st.write("답변: ", answer)
        else:
            st.write("질문을 입력하세요.")

# 18. Streamlit 앱 실행
def main():
    check_login()

    # 페이지 라우팅
    if not st.session_state['logged_in']:
        page = st.sidebar.selectbox("페이지 선택", ["로그인", "회원가입"])
        if page == "로그인":
            login_page()
        elif page == "회원가입":
            signup_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
