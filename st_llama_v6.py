import contextlib
import io
import json
import os
import random
import sys
import pandas as pd

import streamlit as st
from ollama import generate
from enum import Enum

from streamlit.errors import StreamlitAPIException

import st_dataset_v1 as ds
from evaluate import load
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from nltk.util import ngrams

import nltk


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


# Function to save the session state
def save_state():
    to_be_saved_things = ["assistant_msgs", "user_msgs", "ground_truth_msgs", "amount_of_to_be_processed_paragraphs", "system"]
    state_data = {key: value for key, value in st.session_state.items() if not key.startswith('_') and key in to_be_saved_things}
    json_data = json.dumps(state_data)
    st.download_button(
        label="Download state",
        file_name="session_state.json",
        mime="application/json",
        data=json_data
    )


# Function to load the session state
def load_state(file):
    if file is not None:
        try:
            state_data = json.load(file)
            for key, value in state_data.items():
                st.session_state[key] = value
            st.success("Session state loaded successfully!")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid session state file.")
        except StreamlitAPIException:
            pass


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


clear_cache(False)

# tweak-able parameters for llama3 generate
st.session_state.temperature = 0.97
st.session_state.num_predict = 2048
st.session_state.top_p = 0.9
st.session_state.something_downloadable = False

# construct llama3 prompts
begin_token = "<|begin_of_text|>"
start_token = "<|start_header_id|>"
end_token_role = "<|end_header_id|>"
end_token_input = "<|eot_id|>"

predefined_questions = {
    1: "In the provided text missing words are marked with a minus sign. Insert the missing words. Only respond with the corrected text. Do not add any summarization.",
    2: "Improve your text further!",
    3: "Improve your text further!"
}

debug_namings = {
    "assistant_msgs": ("paragraph", "question"),
    "user_msgs": ("question", ""),
    "ground_truth": ("paragraph", "")
}

debug_input_visibility = {
    "assistant_msgs": (False, False),
    "user_msgs": (False, True),
    "ground_truth": (False, True)
}


def calculate_bleu(reference, candidate, max_n=4):
    def count_ngrams(sentence, n):
        return Counter(ngrams(sentence, n))

    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())

    if len(cand_tokens) == 0:
        return 0

    clip_length = max(1, len(cand_tokens) - max_n + 1)

    clipped_counts = {}
    for n in range(1, max_n + 1):
        ref_ngram_counts = count_ngrams(ref_tokens, n)
        cand_ngram_counts = count_ngrams(cand_tokens, n)
        clipped_counts[n] = sum(min(cand_ngram_counts[ngram], ref_ngram_counts[ngram]) for ngram in cand_ngram_counts)

    brevity_penalty = min(1, np.exp(1 - len(ref_tokens) / len(cand_tokens)))

    geometric_mean = np.exp(np.sum([np.log(clipped_counts[n] / max(1, len(cand_tokens) - n + 1)) for n in range(1, max_n + 1)]) / max_n)

    return brevity_penalty * geometric_mean


def sidebar():
    with st.sidebar:
        st.logo('logo.svg')
        with st.expander("**General Stuff**"):
            # displaying the scraped paragraphs
            st.session_state.repeat_count_per_paragraph = st.number_input("repeat amount for current paragraph", step=1, value=1, min_value=1, max_value=50)
            col1, col2 = st.columns([2, 1])
            type_of_text_to_display = {
                "original": st.session_state.content,
                "removed": st.session_state.gapped_results,
            }
            with col2:
                text_type = st.radio(label="which text to show", options=["original", "removed"])
            with col1:
                paragraph_index = st.number_input(label="Start with this paragraph", step=1, min_value=0, max_value=len(type_of_text_to_display[text_type]) - 1)
            st.write(type_of_text_to_display[text_type][paragraph_index])

        with st.expander("**Visibility**"):
            st.markdown(":gray[**Reduce the amount of text covering the screen**]")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state.show_system_prompt = st.toggle("system", value=True)
            with col2:
                st.session_state.show_user_message = st.toggle("user", value=True)
            with col3:
                st.session_state.show_assistant_message = st.toggle("assistant", value=True)
            if st.session_state.repeat_count_per_paragraph * len(st.session_state.user_msgs) > 8:
                st.session_state.show_system_prompt = False
                st.session_state.show_user_message = False
                st.session_state.show_assistant_message = False

        with st.expander("**Predefined questions**"):
            st.text_area("system 1", key="s1", value="The following text is missing one or multiple words. Your task is to listen to the following tasks. ")

            # number input -> amount of questions with key = "q"+i -> collect questions afterward

            amount_of_questions = st.number_input("amount of questions", step=1, value=1, min_value=1, max_value=50)
            for i in range(1, amount_of_questions + 1):
                try:
                    st.text_area(f"question {i}", key=f"q{i}", value=predefined_questions[i])
                except KeyError:
                    st.text_area(f"question {i}", key=f"q{i}", value="Improve your text further!")

        # predefine user input OwO
        sorted_keys = sorted((key for key in st.session_state if key.startswith("q")), key=lambda x: int(x[1:]))
        st.session_state.user_msgs = [st.session_state[key] for key in sorted_keys]
        st.session_state.response = ""

        with st.expander("**LLM Parameters**"):
            st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:", min_value=0.0, value=0.97, max_value=10.0)
            st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128, value=512, max_value=2048)
            st.session_state.top_p = st.slider("**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often", min_value=0.0, value=0.9, max_value=1.0)
        with st.expander("**Obfuscation Parameters**"):
            st.session_state.mask_rate = st.slider("mask rate", 0.0, 1.0, 0.3)
            st.session_state.seed = st.slider("seed", 0, 128, 69)
            random.seed(st.session_state.seed)
        with st.expander("**Load/Save Session**"):
            if st.button("Save State"):
                save_state()

            # Load state file uploader
            uploaded_file = st.file_uploader("Load State", type="json")
            if uploaded_file is not None:
                if st.button("Load State"):
                    load_state(uploaded_file)

        with st.expander("**Debug**"):
            col1, col2 = st.columns(2)
            option = st.selectbox("Which data do you want to display?", ("assistant_msgs", "user_msgs", "ground_truth", "gapped_results"))
            depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1
            try:
                with col1:
                    column = st.number_input(label=debug_namings[option][0], step=1, min_value=0, disabled=debug_input_visibility[option][0])
                with col2:
                    row = st.number_input(label=debug_namings[option][1], step=1, min_value=0, disabled=debug_input_visibility[option][1])
                if depth(st.session_state[option]) == 1:
                    st.info("only column influences data")
                    st.write(st.session_state[option][int(column)])
                elif depth(st.session_state[option]) == 2:
                    st.write(st.session_state[option][int(column)][int(row)])
            except ValueError:
                st.warning("Generate some text first")
            except IndexError:
                st.warning("Index Error")


class Roles(Enum):
    system: str = f"{start_token}system{end_token_role}\n"
    assistant: str = f"{start_token}assistant{end_token_role}\n"
    user: str = f"{start_token}user{end_token_role}\n"


def system_prompt() -> str:
    return f'{begin_token}{Roles.system.value}{st.session_state.system}{end_token_input}{Roles.user.value}'


def assemble_pre_prompt(idx: int) -> str:
    """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant for travel tips and recommendations<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    What is France's capital?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>
    What can I do there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Paris, the City of Light, offers a romantic getaway with must-see attractions like the Eiffel Tower and Louvre Museum, romantic experiences like river cruises and charming neighborhoods, and delicious food and drink options, with helpful tips for making the most of your trip.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Give me a detailed list of the attractions I should visit, and time it takes in each one, to plan my trip accordingly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt: str = system_prompt()
    prompt += st.session_state.user_msgs[0]
    prompt += end_token_input
    prompt += str(Roles.assistant.value)

    for i in range(st.session_state.count + 1):  # count usually starts with 0 -> range(0) is nothing
        prompt += st.session_state.assistant_msgs[idx][i] if st.session_state.assistant_msgs else ""  # avoiding potential exception
        prompt += end_token_input
        prompt += str(Roles.user.value)

        prompt += st.session_state.user_msgs[i]
        prompt += end_token_input
        prompt += str(Roles.assistant.value)
    return prompt


def stream_response(idx: int, stream=True):
    """
    generating response for the nth assistant where n is idx
    using st.assistant_msgs and st.user_msgs for preprompt
    """
    response = generate(
        model='llama3:instruct',
        prompt=assemble_pre_prompt(idx),
        options={
            'num_predict': st.session_state.num_predict,
            'temperature': st.session_state.temperature,
            'top_p': st.session_state.top_p,
            'stop': ['<EOT>'],
        },
        stream=stream
    )
    if stream:
        # yield response
        for chunk in response:
            st.session_state.response += chunk.get("response")
            yield chunk.get("response")
            if chunk.get("done"):
                st.session_state.has_finished = True
                if chunk.get("done_reason") == "length":
                    st.warning("Please increase the LLM Parameter Max tokens")
    else:
        st.session_state.response = response.get("response")
        st.session_state.has_finished = True
        return response.get("response")


def plot_scores(all_bleu_scores, all_bert_scores, all_meteor_scores, index_trend=0):
    # Determine the number of sublists (assuming all score lists have the same structure)
    num_sublists = len(all_bleu_scores)

    # Create the Plotly figure
    fig = go.Figure()
    colors = ['orange', 'red', 'black']

    # Names for each dataset
    dataset_names = ['bleu', 'bert', 'meteor']

    # Add box plots for each question and dataset
    for i, (d1, d2, d3) in enumerate(zip(all_bleu_scores, all_bert_scores, all_meteor_scores), 1):
        for j, (data, color, name) in enumerate(zip([d1, d2, d3], colors, dataset_names)):
            fig.add_trace(go.Box(
                y=data,
                name=f'Q{i}',
                legendgroup=name,
                legendgrouptitle_text=name,
                marker_color=color,
                boxpoints='all',  # Show all points
                offsetgroup=j  # Group boxes for each dataset
            ))

    # Add line plots for the first element of each sublist
    for data, color, name in zip([all_bleu_scores, all_bert_scores, all_meteor_scores], colors, dataset_names):
        first_elements = [sublist[index_trend] for sublist in data]
        fig.add_trace(go.Scatter(
            x=[f'Q{i}' for i in range(1, len(data) + 1)],
            y=first_elements,
            mode='lines+markers',
            name=f'{name} (First Elements)',
            line=dict(color=color, dash='solid'),
            marker=dict(symbol='circle', size=10, color=color),
            legendgroup=name
        ))

    fig.update_layout(
        title='BLEU, BERT and METEOR scores grouped with trend of first repetition',
        yaxis_title='Values',
        xaxis_title='Questions',
        boxmode='group',
        legend_title_text='Datasets',
        legend=dict(groupclick="toggleitem")
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


def abbreviation(length: int) -> str:
    if length == 1:
        return "st"
    elif length == 2:
        return "nd"
    else:
        return "th"


def main():
    # title
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    # preprocessing
    html = ds.scrape_webpage("https://www.gutenberg.org/files/701/701-h/701-h.htm#chap01")  # collect_website
    content: list[str] = ds.extract_content(html)  # initial text
    st.session_state.content = ds.process_text(content)  # split into paragraphs
    st.session_state.ground_truth = st.session_state.content  # static
    paragraphs: list[str] = [ds.create_gaps(s, st.session_state.mask_rate) for s in st.session_state.content]  # remove parts based on seed etc.
    st.session_state.gapped_results = paragraphs  # static

    start_computation: bool
    plot_diagram: bool

    sidebar()

    col1, col2 = st.columns(2)

    with col1:
        start_computation = st.button("Start computation")
        if start_computation:
            if st.session_state.repeat_count_per_paragraph > 1:
                paragraph_progress = st.progress(0, text=f"**processing paragraph the {st.session_state.repeat_count_per_paragraph} {abbreviation(st.session_state.repeat_count_per_paragraph)} time**")
            question_progress = st.progress(0, text=f"about to process **{len(st.session_state.user_msgs)} questions**")
    with col2:
        plot_diagram = st.toggle("Plot diagram")

    if plot_diagram:
        print("someone started to plot a diagram")
        try:
            # Calculate BERTScores
            assistant_msgs_size = len(st.session_state.assistant_msgs)

            original_msgs = [st.session_state.ground_truth[0]] * assistant_msgs_size

            # calculate all BERT scores (outer list contains assistant_msgs_idx; inner list contains refinement_idx: [[q1, q2, q3],[q1, q2, q3],[q1, q2, q3]] where [p1, p2, p3])
            all_bert_scores = []
            for answer_index in range(len(st.session_state.user_msgs)):
                current_assistants_nth_questions = [msg[answer_index] for msg in st.session_state.assistant_msgs]  # will be [q1 of paragraph1, q1 of paragraph2, ...]

                # assistant msgs is correct
                # reference stays the same for answers
                # prediction of is the correct question
                with suppress_stderr():
                    all_bert_scores.append(bertscore.compute(predictions=current_assistants_nth_questions, references=original_msgs, lang="en")['f1'])  # compare two sentences (pred and ref)

            all_meteor_scores = []

            for answer_index in range(len(st.session_state.user_msgs)):
                current_assistants_nth_questions = [msg[answer_index] for msg in st.session_state.assistant_msgs]
                all_meteor_scores.append([meteor.compute(predictions=[predicted], references=[ground_truth])['meteor'] for ground_truth, predicted in zip(original_msgs, current_assistants_nth_questions)])

            all_bleu_scores = []
            for answer_index in range(len(st.session_state.user_msgs)):
                current_assistants_nth_questions = [msg[answer_index] for msg in st.session_state.assistant_msgs]
                all_bleu_scores.append([calculate_bleu(original, altered) for original, altered in zip(original_msgs, current_assistants_nth_questions)])

            plot_scores(all_bleu_scores, all_bert_scores, all_meteor_scores)

        except IndexError:
            fig = go.Figure()
            fig.update_layout(
                title='Development of Elements',
                xaxis_title='Index',
                yaxis_title='Value'
            )
            st.plotly_chart(fig)
            st.warning("Press \"Start computation\" first or Load from file")

    # choose paragraph to be repeatedly iterate over
    st.session_state.paragraph = paragraphs[0]

    pre_stop = 0
    for paragraph_repetition in range(st.session_state.repeat_count_per_paragraph):

        # easy way of dynamically updating assistant_msgs size
        try:
            st.session_state.assistant_msgs[paragraph_repetition]
        except IndexError:
            st.session_state.assistant_msgs.append([])

        # increment random seed - and increment by one each repetition
        random.seed(st.session_state.seed+paragraph_repetition)

        if start_computation:
            print("someone started the computation")
            # display progress paragraphs
            if st.session_state.repeat_count_per_paragraph > 1:
                paragraph_progress.progress((paragraph_repetition + 1) / st.session_state.repeat_count_per_paragraph, text=f"processed {paragraph_repetition + 1} paragraph{"s" if paragraph_repetition > 0 else ""}")

            # system prompt assembly
            st.session_state.system = st.session_state.s1 + "**" + st.session_state.paragraph + "**\n"  # add user prompt and system message
            if st.session_state.show_system_prompt:
                with st.chat_message("system", avatar="./images/system.svg"):
                    st.subheader("system prompt")
                    st.write(st.session_state.system)
            st.session_state.count = -1

        for question_number in range(len(st.session_state.user_msgs)):
            if start_computation:

                # some crazy meme fits here - who needs if else =? if not if :)
                if st.session_state.has_finished:  # completed once
                    st.session_state.count += 1  # using this to access current position in msgs arrays

                    try:
                        st.session_state.user_msgs[st.session_state.count]
                    except IndexError:
                        st.session_state.user_msgs.append("")

                    for assistant in st.session_state.assistant_msgs:
                        try:
                            assistant[st.session_state.count]
                        except IndexError:
                            assistant.append("")

                    st.session_state.has_finished = False

                # generate text for nth passage entry
                if not st.session_state.has_finished:
                    if st.session_state.show_user_message:
                        with st.chat_message("user"):
                            st.write(st.session_state.user_msgs[st.session_state.count])
                    if st.session_state.show_assistant_message:
                        with st.chat_message("assistant"):
                            st.write(stream_response(paragraph_repetition))
                    else:
                        st.write(stream_response(paragraph_repetition, False))

                    question_progress.progress((st.session_state.count + 1) / len(st.session_state.user_msgs), text=f"processed {st.session_state.count + 1} question{"s" if st.session_state.count > 0 else ""}")

                if st.session_state.has_finished:
                    st.session_state.assistant_msgs[paragraph_repetition][st.session_state.count] = st.session_state.response
                    st.session_state.response = ""


if __name__ == "__main__":
    main()
