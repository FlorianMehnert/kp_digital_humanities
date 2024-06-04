import streamlit as st
from process_text import delete_random_words, limit_string_length
def main():

    uploaded_file_left = st.file_uploader("Choose a text file (left)...", type=["txt"])

    if uploaded_file_left is not None:
        # Read the contents of the .txt file and display it in Streamlit
        content_left = uploaded_file_left.read().decode("utf-8").strip()
        #content_left = limit_string_length(content_left, 1000)
        content_left = delete_random_words(content_left, len(content_left.split())/10)

        #stx.scrollableTextbox(content_left, height=300)
        st.text(content_left)

if __name__ == "__main__":
    main()
