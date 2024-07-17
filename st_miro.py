import streamlit as st
import streamlit.components.v1 as components


def main():
    st.set_page_config(page_title="Miro Board Embed", layout="wide")

    st.title("Group 5 of the Collab with digital humanities TUD")
    # Replace 'YOUR_MIRO_BOARD_ID' with the actual ID of your Miro board
    miro_url = "https://miro.com/app/live-embed/uXjVKCsoMCc=/?moveToViewport=7054,8425,21869,12421&embedId=6694599877"
    miro_board_id = 'YOUR_MIRO_BOARD_ID'

    # Set the height based on fullscreen toggle
    height = 700
    components.iframe(miro_url, height=height, scrolling=True)


if __name__ == "__main__":
    main()
