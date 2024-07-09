import pandas as pd
import evaluate
import streamlit as st
import plotly.express as px

# Assume this is the DataFrame exported after interacting with the Llama Chatbot
data = {
    "Original_Text": ["This is the original text."],
    "Text_to_Fill": ["This is the ____ text."],
    "LLM_Repaired_Text": ["This is the repaired text."]
}
df = pd.DataFrame(data)

# List of evaluation metrics
metrics = ["bleu", "bertscore", "meteor"]

# Dictionary to store evaluation results
results = {}

# Calculate scores for each evaluation metric
for metric in metrics:
    evaluator = evaluate.load(metric)
    predictions = df["LLM_Repaired_Text"].tolist()
    references = [[ref] for ref in df["Original_Text"].tolist()]  # Wrap original texts in a list

    if metric == "bertscore":
        result = evaluator.compute(predictions=predictions, references=references, lang="en")
    else:
        result = evaluator.compute(predictions=predictions, references=references)

    results[metric] = result

# Add scores to DataFrame
df['BLEU_Score'] = results['bleu']['bleu']
df['BERTScore_F1'] = results['bertscore']['f1']
df['METEOR_Score'] = results['meteor']['meteor']

# Display DataFrame
print(df)

# Plotting scores in the specified order: BLEU, BERTScore, METEOR
scores = px.bar([df['BLEU_Score'].iloc[0], df['BERTScore_F1'].iloc[0], df['METEOR_Score'].iloc[0]])

st.plotly_chart(scores, use_container_width=False)

# animals=['giraffes', 'orangutans', 'monkeys']

# fig = go.Figure(data=[
#    go.Bar(name='SF Zoo', x=animals, y=[20, 14, 23]),
#    go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])
# ])
# fig.update_layout(barmode='group')
# fig.show()
