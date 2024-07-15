import time

import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from collections import Counter
from nltk.util import ngrams
import nltk
from evaluate import load
from streamlit import cache_resource as cache_resource
import plotly.io as pio
import base64
import streamlit as st

from cuda_stuffs import update_cuda_stats_at_progressbar


def plot_scores(all_bleu_scores, all_meteor_scores, all_bert_scores, index_trend=0, show_trend=False):
    # Determine the number of sublists (assuming all score lists have the same structure)

    # Create the Plotly figure
    fig = go.Figure()
    colors = ['orange', 'red', 'black']

    # Names for each dataset
    dataset_names = ['bleu', 'meteor', 'bert']

    # Add box plots for each question and dataset
    for i, (d1, d2, d3) in enumerate(zip(all_bleu_scores, all_meteor_scores, all_bert_scores), 1):
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

    if show_trend:
        # Add line plots for the first element of each sublist
        for data, color, name in zip([all_bleu_scores, all_meteor_scores, all_bert_scores], colors, dataset_names):
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
        #title='BLEU, METEOR and BERT scores grouped with trend of first repetition',
        yaxis_title='Values',
        xaxis_title='Questions',
        boxmode='group',
        legend_title_text='Datasets',
        legend=dict(groupclick="toggleitem"),
        yaxis_range=[0, 1]
    )

    return fig


def save_as_image(fig, filename="diagram1"):
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image(f"images/{filename}.png", format="png")


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


# Load the metrics
@cache_resource
def load_metrics():
    return load("bertscore"), load("meteor")


def plot_to_png(fig):
    img_bytes = pio.to_image(fig, format="png")
    encoding = base64.b64encode(img_bytes).decode()
    return encoding


def draw_whole_diagram_area():
    try:
        print(f"someone plotted at {time.ctime()}")
        # Calculate scores
        assistant_msgs_size = len(st.session_state.assistant_msgs)

        # placeholders
        original_msgs = [st.session_state.ground_truth[0]] * assistant_msgs_size
        all_bert_scores = []
        all_meteor_scores = []
        all_bleu_scores = []

        for answer_index in range(len(st.session_state.user_msgs)):
            # update cuda progressbar
            update_cuda_stats_at_progressbar()

            current_assistants_nth_questions = [msg[answer_index] for msg in st.session_state.assistant_msgs]
            bertscore, meteor = load_metrics()
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
        st.toast("error")
