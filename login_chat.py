import os
from dotenv import load_dotenv
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
            st.session_state["current_page"] = "챗봇"  # 로그인 성공 후 챗봇 페이지로 이동
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
    # 여기에 챗봇 기능 추가 가능
    st.write("여기에 챗봇과 대화하는 기능을 추가할 수 있습니다.")

# Main 함수에서 로그인 상태에 따라 페이지를 전환
def main():
    check_login()

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "로그인"

    if not st.session_state["logged_in"]:
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
