import streamlit as st
from ollama import generate
from enum import Enum

system = "You are an assistant. You try to find characters that do not belong in the given sentence. Only respond with the corrected sentence. Do not add any summarization."

# Initialize session state variables
if 'has_finished' not in st.session_state:
    st.session_state.has_finished = False
if 'amount_responses' not in st.session_state:
    st.session_state.amount_responses = 3
if 'response' not in st.session_state:
    st.session_state.response = ["" for _ in range(st.session_state.amount_responses)]  # current responses
if 'all_user_messages' not in st.session_state:
    st.session_state.all_user_messages = []  # [['' for _ in range(1)] for _ in range(st.session_state.amount_responses)]
if 'all_assistant_messages' not in st.session_state:
    st.session_state.all_assistant_messages = [[], [], []]  # [['' for _ in range(1)] for _ in range(st.session_state.amount_responses)]
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
if 'disallow_multi_conversation' not in st.session_state:
    st.session_state.disallow_multi_conversations = False
if 'system' not in st.session_state:
    st.session_state.system = system
if 'amount_of_inputs' not in st.session_state:
    st.session_state.amount_of_inputs = 0
if 'disable_amount_responses' not in st.session_state:
    st.session_state.disable_amount_responses = False

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


class Roles(Enum):
    system: str = f"{start_token}system{end_token_role}\n"
    assistant: str = f"{start_token}assistant{end_token_role}\n"
    user: str = f"{start_token}user{end_token_role}\n"


class Prompt():
    def __init__(self, system):
        self.begin_token = "<|begin_of_text|>"
        self.start_token = "<|start_header_id|>"
        self.end_token_role = "<|end_header_id|>"
        self.end_token_input = "<|eot_id|>"
        self.system = system

    def system_prompt(self):
        return f'{begin_token}{Roles.system}{self.system}{end_token_input}{Roles.user}'

    def set_system(self, system: str):
        self.system = system

    def assember_pre_prompt(self, idx):
        prompt: str = self.system_prompt()
        for i in range(st.session_state.amount_of_inputs):
            prompt += st.session_state.all_user_messages[i]
            prompt += end_token_input
            prompt += str(Roles.assistant)  # corresponding to the next message type

            prompt += st.session_state.all_assistant_messages[idx][i]
            prompt += end_token_input
            prompt += str(Roles.user)
        return prompt
def system_prompt() -> str:
    return f'{begin_token}{Roles.system}{st.session_state.system}{end_token_input}{Roles.user}'


def assemble_pre_prompt(idx: int) -> str:
    # get all messages in order for current conversation thread
    prompt: str = system_prompt()
    for i in range(st.session_state.amount_of_inputs):
        prompt += st.session_state.all_user_messages[i]
        prompt += end_token_input
        prompt += str(Roles.assistant)  # corresponding to the next message type

        prompt += st.session_state.all_assistant_messages[idx][i]
        prompt += end_token_input
        prompt += str(Roles.user)
    return prompt


def stream_response(idx: int):
    response = generate(
        model='llama3:instruct',
        # prompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{st.session_state.system}<|eot_id|><|start_header_id|>user<|end_header_id|>{st.session_state.prompt}<|start_header_id|>assistant<|end_header_id|>',
        prompt=assemble_pre_prompt(idx) + st.session_state.prompt + end_token_input + str(Roles.assistant),
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
    if st.session_state.amount_responses > 10:
        st.image(
            'https://i.kym-cdn.com/entries/icons/original/000/000/043/dg1.jpg')
    st.session_state.system = st.text_area("System prompt (e.g. preprompt - instructions)", key="system_input", value=system)
    st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:", min_value=0.0, value=0.97, max_value=10.0)
    st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128, value=128, max_value=2048)
    st.session_state.top_p = st.slider("**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often", min_value=0.0, value=0.9, max_value=1.0)
    if st.session_state.amount_of_inputs > 0 or st.session_state.disable_amount_responses:
        st.session_state.amount_responses = st.slider("amount of responses", min_value=1, value=3, max_value=50, key="response_slider", disabled=True)
    else:
        st.session_state.amount_responses = st.slider("amount of responses", min_value=1, value=3, max_value=50, key="response_slider", disabled=False)
    st.session_state.disallow_multi_conversations = st.toggle("reset converations")  # reset chatter input


# aufbau
# user, eot, role
# assistant, eot, role

def empty_list(type: str) -> list[list[str]]:
    a = []
    for _ in range(st.session_state.amount_responses):
        if type == "assistant":
            a.append([])
    return a


def disable_altering():
    st.session_state.disable_amount_responses = True


def main():
    st.logo("https://ollama.com/public/ollama.png")
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    if st.session_state.disallow_multi_conversations:
        st.session_state.response = ["" for _ in range(st.session_state.amount_responses)]
        st.session_state.all_user_messages = empty_list("user")
        st.session_state.all_assistant_messages = empty_list("assistant")
        st.session_state.amount_of_inputs = 0
        st.session_state.disable_amount_responses = False

    st.session_state.prompt = st.chat_input(placeholder="Your message", key="prompt_input", disabled=False, on_submit=disable_altering)

    # show previous responses like
    # print chat message, print #amount of responses from same index repeat
    for i in range(st.session_state.amount_of_inputs):  # corresponds to first up to previous
        with st.chat_message(name="user", avatar="user"):
            st.write(st.session_state.all_user_messages[i])
        for j in range(st.session_state.amount_responses):
            with st.chat_message(name="assistant", avatar="assistant"):
                st.write(st.session_state.all_assistant_messages[j][i])

    if st.session_state.prompt:
        with st.chat_message(name="assistant", avatar="user"):
            st.write(st.session_state.prompt)
        for i in range(st.session_state.amount_responses):
            with st.chat_message(name="assistant", avatar="assistant"):
                st.write(stream_response(i))
            print("appending:", st.session_state.response[i])
            st.session_state.all_assistant_messages[i].append(st.session_state.response[i])  # increase size by one and fill with response

            st.session_state.something_downloadable = True
        st.session_state.all_user_messages.append(st.session_state.prompt)
        st.session_state.response = ["" for _ in range(st.session_state.amount_responses)]  # empty responses
        st.session_state.amount_of_inputs += 1

    if st.session_state.something_downloadable:
        st.download_button("download responses", "\n\n".join([assemble_pre_prompt(i) for i in range(st.session_state.amount_responses)]))


main()
