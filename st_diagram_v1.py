import plotly.graph_objects as go
import os


def plot_scores(all_bleu_scores, all_bert_scores, all_meteor_scores, index_trend=0, show_trend=False):
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

    if show_trend:
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

    return fig


def save_as_image(fig, filename="diagram1"):
    if not os.path.exists("images"):
        os.mkdir("images")

