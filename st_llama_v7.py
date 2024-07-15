import contextlib
import io
import os
import random
import sys
import time

import nltk
import streamlit as st
from _plotly_utils.exceptions import PlotlyError
from safetensors import torch

from cuda_stuffs import update_cuda_stats_at_progressbar
from data_processing import load_and_process_data, create_gapped_paragraphs
from diagram_utils import load_metrics, draw_whole_diagram_area
from llm_processing import process_llm_responses
from ui_components import create_sidebar, create_main_buttons, create_progress_bars, load_state


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
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.97
    if 'num_predict' not in st.session_state:
        st.session_state.num_predict = 512
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.9
    if 'mask_rate' not in st.session_state:
        st.session_state.mask_rate = 0.3
    if 'seed' not in st.session_state:
        st.session_state.seed = 69
    if 'disabled' not in st.session_state:
        st.session_state.disabled = False
    if 'vram_empty' not in st.session_state:
        st.session_state.vram_empty = None
    if 'BLEU' not in st.session_state:
        st.session_state.BLEU = []
    if 'BERT' not in st.session_state:
        st.session_state.BERT = []
    if 'METEOR' not in st.session_state:
        st.session_state.METEOR = []
    if 'load_state' not in st.session_state:
        st.session_state.load_state = False
    if 'to_be_loaded_state' not in st.session_state:
        st.session_state.to_be_loaded_state = None


clear_cache(False)

# tweak-able parameters for llama3 generate
st.session_state.temperature = 0.97
st.session_state.num_predict = 2048
st.session_state.top_p = 0.9
st.session_state.something_downloadable = False


def abbreviation(length: int) -> str:
    if length == 1:
        return "st"
    elif length == 2:
        return "nd"
    else:
        return "th"


def main():
    # Load metrics
    load_metrics()

    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    # Create UI components
    create_sidebar()
    random.seed(st.session_state.seed)
    start_computation = create_main_buttons()

    # Load and process data
    st.session_state.content = load_and_process_data()
    st.session_state.ground_truth = st.session_state.content
    st.session_state.gapped_results = create_gapped_paragraphs(st.session_state.content, st.session_state.mask_rate)

    if st.session_state.gapped_results:
        st.session_state.paragraph = st.session_state.gapped_results[0]
    else:
        st.error("Empty dataset")
        st.stop()

    if start_computation:
        paragraph_progress, question_progress, time_placeholder, estimate_placeholder = create_progress_bars()
        update_cuda_stats_at_progressbar()

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
            process_llm_responses(
                paragraph_repetition,
                st.session_state.user_msgs,
                st.session_state.assistant_msgs,
                st.session_state.paragraph,
                question_progress
            )

            # Update time estimates
            if paragraph_repetition == 0:
                first_iteration_time = time.time() - st.session_state.start_time
                st.session_state.estimated_total_time = first_iteration_time * len(st.session_state.user_msgs) * st.session_state.repeat_count_per_paragraph
                estimate_placeholder.text(f"Estimated total time: {st.session_state.estimated_total_time:.2f} seconds")
            else:
                elapsed_time = time.time() - st.session_state.start_time
                estimate_placeholder.text(f"Estimated total time: {st.session_state.estimated_total_time:.2f} seconds")
                time_placeholder.text(f"Elapsed time: {elapsed_time:.2f} s")

    # always trying to plot the diagram on update
    try:
        if st.session_state.load_state:
            load_state(st.session_state.to_be_loaded_state)

    except Exception as e:
        st.toast(e)

    try:
        fig = draw_whole_diagram_area()
    except Exception as e:
        print(e, "emptied cuda cache as result of draw whole diagram area")
        st.toast("auto empty cuda cache")
        torch.cuda.empty_cache()
        update_cuda_stats_at_progressbar()
    try:
        st.plotly_chart(fig)
    except PlotlyError:
        print("plotly error")


if __name__ == "__main__":
    main()
