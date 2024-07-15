import plotly.graph_objects as go
import os
import numpy as np
from collections import Counter
from nltk.util import ngrams
import nltk
from evaluate import load
from streamlit import cache_resource as cache_resource


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
        yaxis_range=[0,1]
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
