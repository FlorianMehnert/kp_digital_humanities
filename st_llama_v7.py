import contextlib
import io
import os
import random
import sys
import time

import nltk
import pandas as pd
import streamlit as st

from cuda_stuffs import update_cuda_stats_at_progressbar
from data_processing import load_and_process_data, create_gapped_paragraphs
from diagram_utils import plot_scores, load_metrics, calculate_bleu, plot_to_png
from llm_processing import process_llm_responses
from ui_components import create_sidebar, create_main_buttons, create_progress_bars


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
    bertscore, meteor = load_metrics()

    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    # Create UI components
    create_sidebar()
    random.seed(st.session_state.seed)
    start_computation= create_main_buttons()

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



        try:
            print(f"someone plotted at {time.ctime()}")
            # Calculate scores
            assistant_msgs_size = len(st.session_state.assistant_msgs)
            original_msgs = [st.session_state.ground_truth[0]] * assistant_msgs_size

            all_bert_scores = []
            all_meteor_scores = []
            all_bleu_scores = []

            for answer_index in range(len(st.session_state.user_msgs)):
                # update cuda progressbar
                update_cuda_stats_at_progressbar()

                current_assistants_nth_questions = [msg[answer_index] for msg in st.session_state.assistant_msgs]

                all_bert_scores.append(bertscore.compute(predictions=current_assistants_nth_questions, references=original_msgs, lang="en")['f1'])
                all_meteor_scores.append([meteor.compute(predictions=[predicted], references=[ground_truth])['meteor'] for ground_truth, predicted in zip(original_msgs, current_assistants_nth_questions)])
                all_bleu_scores.append([calculate_bleu(original, altered) for original, altered in zip(original_msgs, current_assistants_nth_questions)])

            st.session_state.BLEU = all_bleu_scores
            st.session_state.BERT = all_bert_scores
            st.session_state.METEOR = all_meteor_scores
            # Plot scores
            st.subheader("BLEU, METEOR and BERT scores")
            st.session_state.fig = plot_scores(all_bleu_scores, all_meteor_scores, all_bert_scores)
            st.plotly_chart(st.session_state.fig)
            st.download_button(
                label="Download PNG",
                data=plot_to_png(st.session_state.fig),
                file_name=f"diagram_r{st.session_state.repeat_count_per_paragraph}_q{len(st.session_state.user_msgs)}.png",
                mime="image/png",
                use_container_width=True
            )

            for _ in range(len(st.session_state.user_msgs)):
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

                to_be_saved = ["system", "ground_truth", "gapped_results", "assistant_msgs", "user_msgs", "BLEU", "BERT", "METEOR"]
                try:
                    settings_to_download = {k: v for k, v in st.session_state.items() if k in to_be_saved}
                    df = pd.DataFrame([(k, v if k == 'system' else v[0] if isinstance(v[0], str) else v[0][0] if len(v[0]) > 0 else v[0])
                                       for k, v in settings_to_download.items()],
                                      columns=['Setting', 'Value'])

                    # Define the desired sort order and new names
                    sort_order = ['ground_truth', 'gapped_results', 'system', 'user_msgs', 'assistant_msgs', 'BLEU', 'METEOR', 'BERT']
                    new_names = {
                        'gapped_results': 'incomplete',
                        'ground_truth': 'ground truth',
                        'system': 'system message',
                        'user_msgs': 'question',
                        'assistant_msgs': 'LLM response'
                    }

                    # Create a categorical column with the specified order
                    df['SortOrder'] = pd.Categorical(df['Setting'], categories=sort_order, ordered=True)

                    # Sort the DataFrame
                    df_sorted = df.sort_values('SortOrder')

                    # Replace \- with - in the gapped_results row and system message
                    df_sorted['Value'] = df_sorted['Value'].replace('\\\\-', '-', regex=True)

                    # Rename the settings
                    df_sorted['Setting'] = df_sorted['Setting'].replace(new_names)

                    # Drop the SortOrder column
                    df_sorted = df_sorted.drop('SortOrder', axis=1)

                    # Reset the index
                    df_sorted = df_sorted.reset_index(drop=True)

                    st.subheader("Responses as dataframe")
                    st.dataframe(df_sorted)
                except Exception as e:
                    st.warning(e)
        except IndexError:
            pass


if __name__ == "__main__":
    main()
