import streamlit as st
from ollama import generate

system = "You are an assistant. You try to find characters that do not belong in the given sentence. Only respond with the corrected sentence. Do not add any summarization."
st.session_state.response = ["", "", ""]
# Initialize session state variables
if 'has_finished' not in st.session_state:
    st.session_state.has_finished = False
if 'response' not in st.session_state:
    st.session_state.response = ["" for _ in range(st.session_state.amount_responses)]
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
st.session_state.system = system
st.session_state.temperature = 0.97
st.session_state.num_predict = 128
st.session_state.top_p = 0.9
st.session_state.something_downloadable = False


def stream_response(prompt: str, idx: int):
    response = generate(
        model='llama3:instruct',
        prompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{st.session_state.system}<|eot_id|><|start_header_id|>user<|end_header_id|>{st.session_state.prompt}<|start_header_id|>assistant<|end_header_id|>',
        options={
            'num_predict': st.session_state.num_predict,
            'temperature': st.session_state.temperature,
            'top_p': st.session_state.top_p,
            'stop': ['<EOT>'],
        },
        stream=True
    )
    for chunk in response:
        st.session_state.response[idx] += chunk.get("response")
    st.session_state.has_finished = True
    yield "END"


# Using "with" notation
with st.sidebar:
    # st.download_button(
    #     label="Download data",
    #     data=st.session_state.response,
    #     file_name="large_df.txt",
    #     mime="text",
    # )

    st.session_state.system = st.text_area("System prompt (e.g. preprompt - instructions)",
                                           key="system_input", value=system)
    st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:",
                                             min_value=0.0,
                                             value=0.97, max_value=10.0)
    st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128,
                                             value=128, max_value=2048)
    st.session_state.top_p = st.slider(
        "**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often",
        min_value=0.0,
        value=0.9, max_value=1.0)
    st.session_state.amount_responses = st.slider("amount of responses", min_value=1,
                                                  value=3, max_value=50)

    st.session_state.response = ["" for _ in range(st.session_state.amount_responses)]
    if st.session_state.amount_responses > 10:
        st.image(
            'https://i.kym-cdn.com/entries/icons/original/000/000/043/dg1.jpg')


def main():
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    # Text area for prompt input
    st.session_state.prompt = st.chat_input(placeholder="Your message", key="prompt_input", disabled=False)
    if st.session_state.prompt:
        with st.spinner("Generating response..."):
            st.session_state.response = ["" for _ in range(st.session_state.amount_responses)]  # Clear previous responses
            for i in range(st.session_state.amount_responses):
                for chunk in stream_response(st.session_state.prompt, i):
                    st.subheader(f"Response{i + 1}")
                    st.write(f"{st.session_state.response[i]}", key=f"response_{i}")
                    if chunk == "END":
                        st.divider()
                        break
            st.session_state.something_downloadable = True
    if st.session_state.something_downloadable:
        st.download_button("download responses", "\n\n".join(st.session_state.response))


# Run the main function
main()
