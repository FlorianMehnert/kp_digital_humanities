import streamlit as st
from ollama import generate
from enum import Enum

system = "You are an assistant. You try to find characters that do not belong in the given sentence. Only respond with the corrected sentence. Do not add any summarization."

st.session_state.response = ["", "", ""]

# Initialize session state variables
if 'has_finished' not in st.session_state:
    st.session_state.has_finished = False
if 'response' not in st.session_state:
    st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
st.session_state.system = system

# tweak-able parameters for llama3 generate
st.session_state.temperature = 0.97
st.session_state.num_predict = 128
st.session_state.top_p = 0.9
st.session_state.something_downloadable = False

# construct llama3 prompts
begin_token = "<|begin_of_text|>"
start_token = "<|start_header_id|>"
end_token_role = "<|end_header_id|>"
end_token_input = "<|eot_id|>"

# keep history
st.session_state.user_msgs = []
st.session_state.assistant_msgs = []


class Roles(Enum):
    system = f"{start_token}system{end_token_role}"
    assistant = f"{start_token}assistant{end_token_role}"
    user = f"{start_token}user{end_token_role}"


def stream_response(idx: int):
    response = generate(
        model='llama3:instruct',
        # prompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{st.session_state.system}<|eot_id|><|start_header_id|>user<|end_header_id|>{st.session_state.prompt}<|start_header_id|>assistant<|end_header_id|>',
        prompt=f'{begin_token}{Roles.system}'
               f'{st.session_state.system}{end_token_input}{Roles.user}'
               f'{st.session_state.prompt}{end_token_input}{Roles.assistant}',
        options={
            'num_predict': st.session_state.num_predict,
            'temperature': st.session_state.temperature,
            'top_p': st.session_state.top_p,
            'stop': ['<EOT>'],
        },
        stream=True
    )
    # yield response
    for chunk in response:
        st.session_state.response[idx] += chunk.get("response")
        yield chunk.get("response")



with st.sidebar:
    st.session_state.system = st.text_area("System prompt (e.g. preprompt - instructions)", key="system_input", value=system)
    st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:", min_value=0.0, value=0.97, max_value=10.0)
    st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128, value=128, max_value=2048)
    st.session_state.top_p = st.slider("**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often", min_value=0.0, value=0.9, max_value=1.0)
    st.session_state.amount_of_responses = st.slider("amount of responses", min_value=1, value=3, max_value=50)

    st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]
    if st.session_state.amount_of_responses > 10:
        st.image(
            'https://i.kym-cdn.com/entries/icons/original/000/000/043/dg1.jpg')


def main():
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    st.session_state.prompt = st.chat_input(placeholder="Your message", key="prompt_input", disabled=False)
    if st.session_state.prompt:
        st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]
        with st.chat_message(name="assistant", avatar="user"):
            st.write(st.session_state.prompt)
        for i in range(st.session_state.amount_of_responses):
            with st.chat_message(name="assistant", avatar="assistant"):
                st.write(stream_response(i))


            # apply previous messages for each response itself - keep 10 different dialogs with amount_responses = 10
            st.session_state.user_msgs.append(st.session_state.prompt)
            st.session_state.user_msgs.append(st.session_state.response)
            st.session_state.something_downloadable = True
    if st.session_state.something_downloadable:
        st.download_button("download responses", "\n\n".join(st.session_state.response))


# Run the main function
main()
