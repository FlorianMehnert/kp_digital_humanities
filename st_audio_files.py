from enum import Enum

import streamlit as st
import requests
from bs4 import BeautifulSoup
import io

if "session" not in st.session_state:
    st.session_state.session = None

if "levels" not in st.session_state:
    st.session_state.levels = {
        "A1": "starter",
        "A2": "elementary1",
        "A2+": "elementary2"
    }
if "books" not in st.session_state:
    st.session_state.books = {
        "Katsudo": "a",
        "Rikai": "c"
    }

if "level" not in st.session_state:
    st.session_state.level = "elementary2"

if "book" not in st.session_state:
    st.session_state.book = "c"

if "lesson" not in st.session_state:
    st.session_state.lesson = 1


def login(email, password):
    session = requests.Session()

    login_url = "https://marugoto.jpf.go.jp/en/form/login"
    response = session.get(login_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    csrf_token = soup.find('input', {'name': '_csrfToken'})['value']
    token = soup.find('input', {'name': 'token'})['value']

    login_data = {
        '_csrfToken': csrf_token,
        'token': token,
        'email': email,
        'password': password
    }

    login_post_url = "https://marugoto.jpf.go.jp/en/form/login"
    response = session.post(login_post_url, data=login_data)

    if "Download Materials" in response.text or "My Account" in response.text:
        return session
    else:
        return None


def filter_audio_list_items(soup):
    def has_audio_div(li):
        audio = li.find('audio')
        if audio:
            return audio and audio.has_attr('src') and audio['src'].endswith('.mp3')
        return False

    list_items = soup.find_all('li', class_='elm_list_item')
    audio_list_items = [li.get_text() for li in list_items if has_audio_div(li)]

    return audio_list_items


def get_audio_sources(session, level: str, book: str, lesson: int):
    audio_page_url = f"https://marugoto.jpf.go.jp/en/download/play/{level}_{book}/lesson{lesson}/"
    response = session.get(audio_page_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    audio_elements = soup.find_all('audio')
    sources = [audio.get('src') for audio in audio_elements if audio.get('src')]

    return sources, filter_audio_list_items(soup)


def get_audio_content(session, url):
    response = session.get(url)
    return io.BytesIO(response.content)


def main():
    st.title("Marugoto Audio Player")

    email = st.text_input("E-mail Address")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state.session = login(email, password)
        if not st.session_state.session:
            st.error("Login failed. Please check your credentials.")
        else:
            st.success("Login successful!")
    with st.sidebar:
        row1, row2, row3 = st.columns(3)

        with row1:
            st.session_state.level = st.session_state.levels[st.selectbox(label="Sprachniveau", options=["A1", "A2", "A2+"])]
        with row2:
            st.session_state.book = st.session_state.books[st.selectbox(options=["Rikai", "Katsudo"], label="どの本？")]
        with row3:
            st.session_state.lesson = st.number_input("lesson", step=1)

        if st.session_state.session:
            audio_sources, audio_names = get_audio_sources(st.session_state.session, st.session_state.level, st.session_state.book, st.session_state.lesson)

            if audio_sources:
                selected_name = st.sidebar.selectbox("Audio Datei auswählen", audio_names)

                selected_source = audio_sources[audio_names.index(selected_name)]
                if selected_source:
                    st.header(selected_name)

                    if selected_source.startswith('/'):
                        base_url = "https://marugoto.jpf.go.jp"
                        full_url = base_url + selected_source
                    else:
                        full_url = selected_source

                    try:
                        audio_content = get_audio_content(st.session_state.session, full_url)
                        st.audio(audio_content, format="audio/mpeg")
                    except Exception as e:
                        st.error(f"Error playing audio: {str(e)}")
                        st.markdown(f"<a href='{full_url}' target='_blank'>Open audio in new tab</a>", unsafe_allow_html=True)

            else:
                st.warning("No audio sources found on the webpage.")


if __name__ == "__main__":
    main()
