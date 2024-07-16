import base64
import streamlit as st
import io
from PIL import Image

# Example base64 string (truncated for brevity)
uploaded_file = st.file_uploader("Choose a file", type=["png"])

# Decode the base64 string

if uploaded_file is not None:
    filename = uploaded_file.name
    try:
        # Read the uploaded file as bytes
        file_bytes = uploaded_file.getvalue()

        # Decode the base64 string
        base64_string = file_bytes.decode('utf-8')
        image_data = base64.b64decode(base64_string)

        # Convert bytes to an image
        image = Image.open(io.BytesIO(image_data))

        # Display the image
        st.image(image, caption='Decoded PNG Image')
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.download_button(
                label="Download PNG",
                data=image_data,
                file_name=filename,
                mime="image/png",
                use_container_width=True
            )
