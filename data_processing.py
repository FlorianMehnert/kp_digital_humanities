import st_dataset_v1 as ds
import streamlit as st


def load_and_process_data(scrape=False):
    """
    either loads the web content or uses the input of a text area as dataset content
    """
    if scrape:
        html = ds.scrape_webpage("https://www.gutenberg.org/files/701/701-h/701-h.htm#chap01")
        content = ds.extract_content(html)
        processed_content = ds.process_text(content)
    else:
        processed_content = [st.session_state.dataset] if "dataset" in st.session_state else []
    return processed_content


def create_gapped_paragraphs(content, mask_rate):

    return [ds.create_gaps(s, mask_rate) for s in content] if  content != [""] else []
