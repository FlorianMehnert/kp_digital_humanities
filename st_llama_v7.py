import contextlib
import io
import os
import random
import sys

import pandas as pd
from evaluate import load
import numpy as np
from collections import Counter
from nltk.util import ngrams

from st_diagram_v1 import plot_scores, save_as_image

import streamlit as st
from data_processing import load_and_process_data, create_gapped_paragraphs
from ui_components import create_sidebar, create_main_buttons, create_progress_bars
from llm_processing import process_llm_responses

import nltk

import time

st.set_page_config(
    page_title="OCR inpainting",
    page_icon="🦙"
)


@contextlib.contextmanager
def suppress_stderr():
    """
    A context manager that redirects stderr to devnull
    """
    stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr




def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


# Download necessary NLTK data
nltk.download('punkt', quiet=True)


# Load the metrics
@st.cache_resource
def load_metrics():
    return load("bertscore"), load("meteor")


bertscore, meteor = load_metrics()

system = "You are an assistant. You try to find characters that do not belong in the given sentence. Only respond with the corrected sentence. Do not add any summarization."


def clear_cache(full_reset=True):
    if full_reset:
        keys = list(st.session_state.keys())
        for key in keys:
            st.session_state.pop(key)

    if 'has_finished' not in st.session_state:
        st.session_state.has_finished = True
    if 'amount_responses' not in st.session_state:
        st.session_state.amount_of_responses = 3
    if 'response' not in st.session_state:
        st.session_state.response = ""  # CURRENT RESPONSE
    if 'user_msgs' not in st.session_state:
        st.session_state.user_msgs = []  # ALL inputs
    if 'assistant_msgs' not in st.session_state:
        st.session_state.assistant_msgs = [[]]  # ALL responses
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""
    if 'disallow_multi_conversation' not in st.session_state:
        st.session_state.disallow_multi_conversations = False
    if 'system' not in st.session_state:
        st.session_state.system = system
    if 'amount_of_inputs' not in st.session_state:
        st.session_state.amount_of_inputs = 0
    if 'count' not in st.session_state:
        st.session_state.count = -1
    if 'option' not in st.session_state:
        st.session_state.option = "default markdown"
    if 'ground_truth' not in st.session_state:
        st.session_state.ground_truth = []
    if 'gapped_results' not in st.session_state:
        st.session_state.gapped_results = []
    if 'bert_scores' not in st.session_state:
        st.session_state.bert_scores = []
    if 'content' not in st.session_state:
        st.session_state.content = []
    if 'repeat_count_per_paragraph' not in st.session_state:
        st.session_state.repeat_count_per_paragraph = 1
    if 'show_system_prompt' not in st.session_state:
        st.session_state.show_system_prompt = True
    if 'show_user_message' not in st.session_state:
        st.session_state.show_user_message = True
    if 'show_assistant_message' not in st.session_state:
        st.session_state.show_assistant_message = True
    if 'mask_rate' not in st.session_state:
        st.session_state.mask_rate = 0.3
    if 'start_time' not in st.session_state:
        st.session_state.start_time = 0
    if 'first_iteration_time' not in st.session_state:
        st.session_state.first_iteration_time = None
    if 'estimated_total_time' not in st.session_state:
        st.session_state.estimated_total_time = None


clear_cache(False)

# tweak-able parameters for llama3 generate
st.session_state.temperature = 0.97
st.session_state.num_predict = 2048
st.session_state.top_p = 0.9
st.session_state.something_downloadable = False


def calculate_bleu(reference, candidate, max_n=4):
    def count_ngrams(sentence, n):
        return Counter(ngrams(sentence, n))

    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())

    if len(cand_tokens) == 0:
        return 0

    max(1, len(cand_tokens) - max_n + 1)

    clipped_counts = {}
    for n in range(1, max_n + 1):
        ref_ngram_counts = count_ngrams(ref_tokens, n)
        cand_ngram_counts = count_ngrams(cand_tokens, n)
        clipped_counts[n] = sum(min(cand_ngram_counts[ngram], ref_ngram_counts[ngram]) for ngram in cand_ngram_counts)

    brevity_penalty = min(1, np.exp(1 - len(ref_tokens) / len(cand_tokens)))

    geometric_mean = np.exp(np.sum([np.log(clipped_counts[n] / max(1, len(cand_tokens) - n + 1)) for n in range(1, max_n + 1)]) / max_n)

    return brevity_penalty * geometric_mean


def abbreviation(length: int) -> str:
    if length == 1:
        return "st"
    elif length == 2:
        return "nd"
    else:
        return "th"


def main():
    # Load metrics
    bertscore, meteor = load_metrics()

    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    # Load and process data
    st.session_state.content = load_and_process_data()
    st.session_state.ground_truth = st.session_state.content
    st.session_state.gapped_results = create_gapped_paragraphs(st.session_state.content, st.session_state.mask_rate)

    # Create UI components
    create_sidebar()
    random.seed(st.session_state.seed)
    start_computation, save_diagram, plot_diagram = create_main_buttons()

    if st.session_state.gapped_results:
        st.session_state.paragraph = st.session_state.gapped_results[0]
    else:
        st.error("Empty dataset")
        st.stop()

    if plot_diagram:
        try:
            # Calculate scores
            assistant_msgs_size = len(st.session_state.assistant_msgs)
            original_msgs = [st.session_state.ground_truth[0]] * assistant_msgs_size

            all_bert_scores = []
            all_meteor_scores = []
            all_bleu_scores = []

            for answer_index in range(len(st.session_state.user_msgs)):
                current_assistants_nth_questions = [msg[answer_index] for msg in st.session_state.assistant_msgs]

                all_bert_scores.append(bertscore.compute(predictions=current_assistants_nth_questions, references=original_msgs, lang="en")['f1'])
                all_meteor_scores.append([meteor.compute(predictions=[predicted], references=[ground_truth])['meteor'] for ground_truth, predicted in zip(original_msgs, current_assistants_nth_questions)])
                all_bleu_scores.append([calculate_bleu(original, altered) for original, altered in zip(original_msgs, current_assistants_nth_questions)])

            # Plot scores
            fig = plot_scores(all_bleu_scores, all_meteor_scores, all_bert_scores)
            st.plotly_chart(fig)
            if save_diagram:
                file_path = f"diagram_r{st.session_state.repeat_count_per_paragraph}_q{len(st.session_state.user_msgs)}"
                save_as_image(fig, file_path)
                with open("images/" + file_path + ".png", 'rb') as file:
                    btn = st.download_button(
                        label="Download PNG",
                        data=file,
                        file_name=file_path,
                        mime="image/png"
                    )
            for i, amsg in enumerate(st.session_state.user_msgs):
                st.markdown(
                    """
                    <style>
                    .stTable {
                        width: 100% !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.dataframe(pd.DataFrame(st.session_state.assistant_msgs[i]), width=None)
        except IndexError:
            st.warning("Press \"Start computation\" first or Load from file")

    if start_computation:
        paragraph_progress, question_progress, time_placeholder, estimate_placeholder = create_progress_bars()

        st.session_state.start_time = time.time()
        for paragraph_repetition in range(st.session_state.repeat_count_per_paragraph):
            # Ensure assistant_msgs has enough elements
            if len(st.session_state.assistant_msgs) <= paragraph_repetition:
                st.session_state.assistant_msgs.append([])
            # Update paragraph progress
            if st.session_state.repeat_count_per_paragraph > 1:
                paragraph_progress.progress(
                    (paragraph_repetition + 1) / st.session_state.repeat_count_per_paragraph,
                    text=f"{paragraph_repetition + 1}{abbreviation(paragraph_repetition + 1)} repetition of {st.session_state.repeat_count_per_paragraph} total repetitions"
                )
            # Process LLM responses
            for question_number in range(len(st.session_state.user_msgs)):
                process_llm_responses(
                    paragraph_repetition,
                    st.session_state.user_msgs,
                    st.session_state.assistant_msgs,
                    st.session_state.paragraph
                )

                # Update question progress
                question_progress.progress(
                    (st.session_state.count + 1) / len(st.session_state.user_msgs),
                    text=f"processed {st.session_state.count + 1} question{'s' if st.session_state.count > 0 else ''}"
                )

                # Update time estimates
                if question_number == 0 and paragraph_repetition == 0:
                    first_iteration_time = time.time() - st.session_state.start_time
                    st.session_state.estimated_total_time = first_iteration_time * len(st.session_state.user_msgs) * st.session_state.repeat_count_per_paragraph
                    estimate_placeholder.text(f"Estimated total time: {st.session_state.estimated_total_time:.2f} seconds")
                else:
                    elapsed_time = time.time() - st.session_state.start_time
                    estimate_placeholder.text(f"Estimated total time: {st.session_state.estimated_total_time:.2f} seconds")
                    time_placeholder.text(f"Elapsed time: {elapsed_time:.2f} s")


if __name__ == "__main__":
    main()
