import sqlite3

import streamlit as st
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import mysql.connector
from mysql.connector import Error


scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

# Database connection
def create_db_connection():
    try:
        connection = sqlite3.connect('songs.db')
        return connection
    except Error as e:
        st.error(f"Error connecting to SQLite: {e}")
        return None


def create_table_if_not_exists(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS song_attributes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            title TEXT NOT NULL,
            attributes TEXT
        )
        """)
        connection.commit()
    except Error as e:
        st.error(f"Error creating table: {e}")


def save_song_attributes(connection, video_id, title, attributes):
    try:
        cursor = connection.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO song_attributes (video_id, title, attributes)
        VALUES (?, ?, ?)
        """, (video_id, title, attributes))
        connection.commit()
        st.success("Attributes saved successfully!")
    except Error as e:
        st.error(f"Error saving attributes: {e}")


def main():
    st.title("YouTube Playlist Viewer and Music Search")

    # Database connection
    db_connection = create_db_connection()
    if db_connection:
        create_table_if_not_exists(db_connection)

    # Authentication
    if 'credentials' not in st.session_state:
        authenticate()
    else:
        menu = ["View Playlists", "Search YouTube Music", "Manage Attributes"]
        choice = st.sidebar.selectbox("Choose an option", menu)

        if choice == "View Playlists":
            show_playlist_contents()
        elif choice == "Search YouTube Music":
            search_youtube_music(db_connection)
        elif choice == "Manage Attributes":
            manage_attributes()


def authenticate():
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        "client_secret.json", scopes)
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

    authorization_url, _ = flow.authorization_url(prompt="consent")

    st.write("Please visit this URL to authorize the application:")
    st.write(authorization_url)

    code = st.text_input("Enter the authorization code:")
    if code:
        flow.fetch_token(code=code)
        st.session_state.credentials = flow.credentials
        st.experimental_rerun()


def show_playlist_contents():
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", credentials=st.session_state.credentials)

    playlists = get_playlists(youtube)
    selected_playlist_title = st.selectbox("Select a playlist:", list(playlists.keys()))

    if selected_playlist_title:
        selected_playlist_id = playlists[selected_playlist_title]
        playlist_items = get_playlist_items(youtube, selected_playlist_id)
        st.write(f"Contents of '{selected_playlist_title}':")
        for item in playlist_items:
            st.write(f"- {item}")


def search_videos(youtube, query):
    request = youtube.search().list(
        q=query,
        type="video",
        part="id,snippet",
        maxResults=10,
        videoCategoryId="10"  # 10 is the category ID for Music
    )
    response = request.execute()
    return [{"id": item['id']['videoId'], "title": item['snippet']['title']} for item in response['items']]


def get_playlists(youtube):
    request = youtube.playlists().list(
        part="snippet",
        mine=True,
        maxResults=50
    )
    response = request.execute()
    return {item['snippet']['title']: item['id'] for item in response['items']}


def get_playlist_items(youtube, playlist_id):
    request = youtube.playlistItems().list(
        part="snippet",
        playlistId=playlist_id,
        maxResults=50
    )
    response = request.execute()
    return [item['snippet']['title'] for item in response['items']]


def search_youtube_music(db_connection):
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", credentials=st.session_state.credentials)

    query = st.text_input("Enter a song or artist name:")
    if query:
        search_results = search_videos(youtube, query)
        if search_results:
            selected_video = st.selectbox("Select a video:", search_results, format_func=lambda x: x['title'])
            if selected_video:
                st.write(f"You selected: {selected_video['title']}")
                st.write(f"Video ID: {selected_video['id']}")
                st.video(f"https://www.youtube.com/watch?v={selected_video['id']}")

                # Attribute management
                if 'attributes' not in st.session_state:
                    st.session_state.attributes = []

                new_attribute = st.text_input("Add a new attribute:")
                if st.button("Add Attribute"):
                    if new_attribute and new_attribute not in st.session_state.attributes:
                        st.session_state.attributes.append(new_attribute)

                selected_attributes = st.multiselect("Select attributes for this song:", st.session_state.attributes)

                if st.button("Save Attributes"):
                    save_song_attributes(db_connection, selected_video['id'], selected_video['title'], str(selected_attributes))
        else:
            st.write("No results found.")


def manage_attributes():
    if 'attributes' not in st.session_state:
        st.session_state.attributes = []

    st.subheader("Manage Attributes")
    new_attribute = st.text_input("Add a new attribute:")
    if st.button("Add"):
        if new_attribute and new_attribute not in st.session_state.attributes:
            st.session_state.attributes.append(new_attribute)
            st.success(f"Added: {new_attribute}")
        else:
            st.warning("Attribute already exists or is empty.")

    st.write("Current Attributes:")
    for attr in st.session_state.attributes:
        st.write(f"- {attr}")


if __name__ == "__main__":
    main()
