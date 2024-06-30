import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
import random

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    return [word for word in words if word.isalnum() and word not in stop_words]


def calculate_tfidf(text1, text2):
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    feature_names = np.array(vectorizer.get_feature_names_out())
    return tfidf_matrix, feature_names


def create_semantic_galaxy(text1, text2):
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    tfidf_matrix, feature_names = calculate_tfidf(" ".join(words1), " ".join(words2))
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    common_words = set(words1) & set(words2)

    def get_word_coords(word, text_index):
        if word in feature_names:
            idx = np.where(feature_names == word)[0][0]
            importance = tfidf_matrix[text_index, idx]
            if hasattr(importance, 'toarray'):
                importance = importance.toarray()[0][0]
            else:
                importance = float(importance)
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0.1, 1)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            return x, y, importance
        return None, None, 0

    traces = []

    for i, words in enumerate([words1, words2]):
        x, y, sizes = [], [], []
        hover_texts = []
        for word in words:
            x_coord, y_coord, importance = get_word_coords(word, i)
            if x_coord is not None:
                x.append(x_coord + i * 2)  # Offset the second galaxy
                y.append(y_coord)
                sizes.append(importance * 50)  # Scale importance for visibility
                hover_texts.append(f"Word: {word}<br>Importance: {importance:.4f}")

        traces.append(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=['blue' if i == 0 else 'red'] * len(x),
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            text=[word if imp > 0.1 else '' for word, imp in zip(words, sizes)],
            textposition="top center",
            hoverinfo='text',
            hovertext=hover_texts,
            name=f'Text {i + 1}'
        ))

    # Add "binary stars" for common words
    x_common, y_common, sizes_common = [], [], []
    hover_texts_common = []
    for word in common_words:
        x1, y1, imp1 = get_word_coords(word, 0)
        x2, y2, imp2 = get_word_coords(word, 1)
        if x1 is not None and x2 is not None:
            x_common.extend([x1, x2 + 2])
            y_common.extend([y1, y2])
            sizes_common.extend([imp1 * 50, imp2 * 50])
            hover_texts_common.extend([f"Common Word: {word}<br>Importance: {imp1:.4f}",
                                       f"Common Word: {word}<br>Importance: {imp2:.4f}"])

    traces.append(go.Scatter(
        x=x_common, y=y_common,
        mode='markers',
        marker=dict(
            size=sizes_common,
            color='yellow',
            opacity=0.8,
            line=dict(width=2, color='orange')
        ),
        hoverinfo='text',
        hovertext=hover_texts_common,
        name='Common Words'
    ))

    layout = go.Layout(
        title="Semantic Galaxy Visualization",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        showlegend=True,
        annotations=[
            dict(
                x=1, y=1,
                xref="paper", yref="paper",
                text=f"Similarity: {similarity:.2f}",
                showarrow=False,
                font=dict(size=16)
            )
        ]
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


# Streamlit app
st.title("Semantic Galaxy: Text Similarity Visualization")

text1 = st.text_area("Enter first text:")
text2 = st.text_area("Enter second text:")

if st.button("Generate Semantic Galaxy"):
    if text1 and text2:
        fig = create_semantic_galaxy(text1, text2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Please enter both texts.")
