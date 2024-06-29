import random

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
import nltk

st.set_page_config(page_title="Dataset Preprocessing", page_icon=":material/travel_explore:")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# st.set_page_config(page_title="data - 🪱", layout="wide")
css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 400px;
        max-width: 800px;
    }
</style>
'''


# Function to scrape the webpage
@st.cache_data(ttl=86400)  # cache for one day lul
def scrape_webpage_v1(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error("Failed to retrieve the webpage.")
        return None


# Function to extract content from the HTML
def extract_content(html) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    # Example: Extract all paragraphs
    paragraphs = soup.find_all('p')
    return [p.get_text() for p in paragraphs]


def load_newspaper_text(to_be_loaded_filename: str) -> str | None:
    """Load preprocessed newspaper text file"""
    temp_text: list[str] = []
    try:
        with open(to_be_loaded_filename, 'r') as f:
            return f.read()

    except FileNotFoundError:
        print(f"File not found: {to_be_loaded_filename}")
        return None


def split_by_paragraph(full_text: str) -> list[str]:
    return full_text.split('\n\n')


def remove_headlines_etc(paragraphs: list[str]) -> list[str]:
    # remove all entries in paragraphs that are markdown headlines
    result: list[str] = [p for p in paragraphs if not p.startswith("#") and not p == "" and not p.isupper()]
    result = [p.strip("\r\n") for p in result]
    return result


def process_text(full_text: str | list[str]) -> list[str]:
    temp_text: list[str]
    if type(full_text) is str:
        temp_text = split_by_paragraph(full_text.strip())
    else:
        temp_text = full_text
    temp_text = remove_headlines_etc(temp_text)
    return temp_text


def limit_string_length(text: str, max_length: int):
    if len(text) > max_length:
        return text[:max_length].rstrip() + '...'
    else:
        return text


def create_gaps(sentence, mask_rate=0.2):
    replacement_token = ""
    if st.session_state.option == "pandas layout":
        replacement_token = "-"
    else:
        replacement_token = "\\-"
    words = sentence.split()
    num_words = len(words)

    # Calculate the total number of words to remove based on mask rate
    total_to_remove = int(num_words * mask_rate)
    if total_to_remove < 2:
        total_to_remove = 2  # Ensure at least two words are removed

    # Ensure we are removing from two different positions
    if total_to_remove > num_words:
        total_to_remove = num_words  # Cap the number of words to be removed to the total number of words available

    # Select two unique starting positions
    positions = sorted(random.sample(range(num_words), 2))

    # set current position to empty string, then march to left or right on random until edge is hit or empty string is hit
    # try again if empty string on the other side
    # if not successful go to second head
    def try_remove(start, num_to_remove) -> int:
        offset = 0
        current_pos: int = start
        remaining_words = num_to_remove
        directions = range(2)  # left is False right is true
        direction = random.choice(directions)
        redirected = 0

        while remaining_words > 0:

            # get current position
            if not direction:
                current_pos = start - offset
            elif direction:
                current_pos = start + offset

            # is in bounds
            if current_pos < len(words) and direction or current_pos >= 0 and not direction:

                if words[current_pos] != replacement_token:
                    words[current_pos] = replacement_token
                    remaining_words -= 1
                offset += 1
            else:
                if redirected <= 1:  # turn into other direction
                    direction = int(not directions.index(direction))
                    redirected += 1
                    offset = 1
                else:
                    return remaining_words
        return 0

    # Distribute the words to remove between the two positions
    split: float = random.random()
    amount_to_remove_1 = int(split * len(words) * mask_rate + 1)
    amount_to_remove_2 = int((1 - split) * len(words) * mask_rate)
    # print("atr", amount_to_remove_1, amount_to_remove_2, "len words", len(words), "split", split, "modifier")

    # Remove words from the first position
    first_pos = positions[0]
    amount_to_remove_2 += try_remove(first_pos, amount_to_remove_1)

    # Remove words from the second position
    second_pos = positions[1]
    leftovers = try_remove(second_pos, amount_to_remove_2)

    # Reconstruct the sentence
    new_sentence = ' '.join(words)

    return new_sentence


def main():
    # define sidebar widget
    with st.sidebar:
        st.session_state.option = st.selectbox(
            "default or pandas 🐼?",
            ("default markdown", "pandas layout"))
        st.session_state.mask_rate = st.slider("mask rate", 0.0, 1.0, 0.3)
        st.session_state.seed = st.slider("seed", 0, 128, 69)
        st.info("the current masking function can remove dots")
    random.seed(st.session_state.seed)
    # collect data
    html = scrape_webpage_v1("https://www.gutenberg.org/files/701/701-h/701-h.htm#chap01")
    content = extract_content(html)
    content = process_text(content)

    # remove random words per paragraph
    gapped_content = [create_gaps(s, st.session_state.mask_rate) for s in content]

    # change layout based on toggle
    if st.session_state.option == "pandas layout":
        data = {
            "Number of Sentences": [len(sent_tokenize(p)) for p in content],  # use language processing to identify sentences
            "Content": content,
            "Gapped content": gapped_content
        }
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        for p, g, i in zip(content, gapped_content, range(len(content))):
            with st.container():
                col1, col2, col3 = st.columns([1, 15, 16])
                col1.write(str(len(sent_tokenize(p))))
                col2.write(p)

                g.replace('-', '\-')
                g.replace('.', '\\.')
                col3.write(g)


if __name__ == '__main__':
    main()
