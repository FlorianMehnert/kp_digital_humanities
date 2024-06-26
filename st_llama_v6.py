import random

import streamlit as st
from ollama import generate
from enum import Enum
import st_dataset_v1 as ds

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
        st.session_state.response = [""]  # CURRENT RESPONSE
    if 'user_msgs' not in st.session_state:
        st.session_state.user_msgs = []  # ALL inputs
    if 'assistant_msgs' not in st.session_state:
        st.session_state.assistant_msgs = [[], [], []]  # ALL responses
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

with st.sidebar:
    st.logo('logo.svg')
    with st.expander("**Predefined questions**"):
        s1 = st.text_area("system 1", key="s1", value="The following text is missing one or multiple words. Your task is to listen to the following tasks. ")
        q1 = st.text_area("question 1", key="q1", value="In the provided text missing words are marked with a minus sign. Insert the missing words. Only respond with the corrected text. Do not add any summarization.")
        q2 = st.text_area("question 2", key="q2", value="Improve your text further!")
        q3 = st.text_area("question 3", key="q3", value="Try to improve on your text!")

    # predefine user input OwO
    sorted_keys = sorted((key for key in st.session_state if key.startswith("q")), key=lambda x: int(x[1:]))
    st.session_state.user_msgs = [st.session_state[key] for key in sorted_keys]
    st.session_state.response = ["" for _ in range(len(st.session_state.user_msgs))]

    with st.expander("**LLM Parameters**"):
        st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:", min_value=0.0, value=0.97, max_value=10.0)
        st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128, value=512, max_value=2048)
        st.session_state.top_p = st.slider("**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often", min_value=0.0, value=0.9, max_value=1.0)
    with st.expander("**Obfuscation Parameters**"):
        st.session_state.mask_rate = st.slider("mask rate", 0.0, 1.0, 0.3)
        st.session_state.seed = st.slider("seed", 0, 128, 69)
        random.seed(st.session_state.seed)


class Roles(Enum):
    system: str = f"{start_token}system{end_token_role}\n"
    assistant: str = f"{start_token}assistant{end_token_role}\n"
    user: str = f"{start_token}user{end_token_role}\n"


def system_prompt() -> str:
    return f'{begin_token}{Roles.system.value}{st.session_state.system}{end_token_input}{Roles.user.value}'


def assemble_pre_prompt(idx: int) -> str:
    # get all messages in order for current conversation thread
    prompt: str = system_prompt()
    for i in range(st.session_state.count + 1):
        prompt += st.session_state.user_msgs[i]
        prompt += end_token_input
        prompt += str(Roles.assistant.value)

        prompt += st.session_state.assistant_msgs[idx][i] if st.session_state.assistant_msgs else ""  # is changing
        prompt += end_token_input
        prompt += str(Roles.user.value)
    return prompt


def stream_response(idx: int):
    """
    generating response for the nth assistant where n is idx
    using st.assistant_msgs and st.user_msgs for preprompt
    """
    response = generate(
        model='llama3:instruct',
        # prompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{st.session_state.system}<|eot_id|><|start_header_id|>user<|end_header_id|>{st.session_state.prompt}<|start_header_id|>assistant<|end_header_id|>',
        prompt=assemble_pre_prompt(idx) + st.session_state.prompt + end_token_input + str(Roles.assistant.value),
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
        if chunk.get("done"):
            st.session_state.has_finished = True
            if chunk.get("done_reason") == "length":
                st.warning("Please increase the LLM Parameter Max tokens")


def main():
    # title
    st.logo("https://ollama.com/public/ollama.png")
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    html = ds.scrape_webpage("https://www.gutenberg.org/files/701/701-h/701-h.htm#chap01")  # collect_website
    content: list[str] = ds.extract_content(html)  # initial text
    content = ds.process_text(content)  # split into paragraphs
    paragraphs: list[str] = [ds.create_gaps(s, st.session_state.mask_rate) for s in content]  # remove parts based on seed etc.

    # buttons
    start_computation = st.button("Start computation")
    pre_stop = 0
    for paragraph, paragraph_number in zip(paragraphs, range(len(paragraphs))):
        if pre_stop > 1:
            break
        pre_stop += 1
        if start_computation:
            st.session_state.system = st.session_state.s1 + "**" + paragraph + "**\n"  # add user prompt and system message
            with st.chat_message("system", avatar="system.svg"):
                st.subheader("system prompt")
                st.write(st.session_state.system)
            st.session_state.count = -1
        for question_number in range(len(st.session_state.user_msgs)):
            if start_computation:
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

                    # appending response from previous iteration
                    st.session_state.assistant_msgs[paragraph_number][st.session_state.count - 1] = st.session_state.response[paragraph_number]

                    st.session_state.response = ["" for _ in range(len(st.session_state.user_msgs))]

                # generate text for nth passage entry
                if not st.session_state.has_finished:
                    with st.chat_message("user"):
                        st.write(st.session_state.user_msgs[st.session_state.count])

                    with st.chat_message("assistant"):
                        st.write(stream_response(st.session_state.count - 1))


if __name__ == "__main__":
    main()
