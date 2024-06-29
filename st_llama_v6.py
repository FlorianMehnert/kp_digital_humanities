import streamlit as st
from ollama import generate
from enum import Enum
import st_dataset_v1

system = "You are an assistant. You try to find characters that do not belong in the given sentence. Only respond with the corrected sentence. Do not add any summarization."

# Initialize session state variables
if 'has_finished' not in st.session_state:
    st.session_state.has_finished = False
if 'amount_responses' not in st.session_state:
    st.session_state.amount_of_responses = 3
if 'response' not in st.session_state:
    st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]  # CURRENT response
if 'user_msgs' not in st.session_state:
    st.session_state.user_msgs = [""]  # ALL inputs
if 'assistant_msgs' not in st.session_state:
    st.session_state.assistant_msgs = [[""], [""], [""]]  # ALL responses
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
if 'count' not in st.session_state:
    st.session_state.count = 0

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

def system_prompt() -> str:
    return f'{begin_token}{Roles.system.value}{st.session_state.system}{end_token_input}{Roles.user.value}'


def assemble_pre_prompt(idx: int) -> str:
    # get all messages in order for current conversation thread
    prompt: str = system_prompt()
    for i in range(st.session_state.count):
        st.write("in preprompt")
        st.write(st.session_state.user_msgs)
        st.write(st.session_state.assistant_msgs)
        st.write("after preprompt")
        prompt += st.session_state.user_msgs[i] if st.session_state.user_msgs[idx] else ""  # is changing
        prompt += end_token_input
        prompt += str(Roles.assistant.value)

        prompt += st.session_state.assistant_msgs[idx][i] if st.session_state.assistant_msgs else "" # is changing
        prompt += end_token_input
        prompt += str(Roles.user.value)
    st.write(prompt)
    return prompt


def stream_response(idx: int):
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


with st.sidebar:
    if st.session_state.amount_of_responses > 10:
        st.image(
            'https://i.kym-cdn.com/entries/icons/original/000/000/043/dg1.jpg')
    st.session_state.system = st.text_area("System prompt (e.g. preprompt - instructions)", key="system_input", value=system)
    st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:", min_value=0.0, value=0.97, max_value=10.0)
    st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128, value=128, max_value=2048)
    st.session_state.top_p = st.slider("**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often", min_value=0.0, value=0.9, max_value=1.0)
    if st.session_state.amount_of_inputs > 0 or st.session_state.disable_amount_responses:
        st.session_state.amount_of_responses = st.slider("amount of responses", min_value=1, value=3, max_value=50, key="response_slider", disabled=True)
    else:
        st.session_state.amount_of_responses = st.slider("amount of responses", min_value=1, value=3, max_value=50, key="response_slider", disabled=False)
    st.session_state.disallow_multi_conversations = st.toggle("reset converations")  # reset chatter input


# aufbau
# user, eot, role
# assistant, eot, role

def empty_list(type: str) -> list[list[str]]:
    a = []
    for _ in range(st.session_state.amount_of_responses):
        if type == "assistant":
            a.append([])
    return a


def disable_altering():
    st.session_state.disable_amount_responses = True


original_passage = ["There is nothing of all this in \"The King of the Golden River.\" Unlike his other works, it was written merely to entertain. Scarcely that, since it was not written for publication at all, but to meet a challenge set him by a young girl."]
gapped_passage = ["- - - - - - - - - - - - - - his other works, it was written merely to entertain. Scarcely that, since it was not written for publication at all, but to meet a challenge set him by a young girl."]
testing_message = ["I am alpha", "I am bert", "I am chirby", "I am dirk", "I am einstein", "I am frank", "I am gerald", "I am heist"]

gapped_passage[0].replace('-', '\-')
gapped_passage[0].replace('.', '\\.')

def main():
    st.logo("https://ollama.com/public/ollama.png")
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")

    if st.session_state.disallow_multi_conversations:
        st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]
        st.session_state.user_msgs = empty_list("user")
        st.session_state.assistant_msgs = empty_list("assistant")
        st.session_state.amount_of_inputs = 0
        st.session_state.disable_amount_responses = False

    col1, col2 = st.columns(2)
    with col1:
        start_computation = st.button("Start computation")
    with col2:
        reset_count = st.button("Reset")
    if start_computation:

        st.session_state.count += 1
        st.write("st count is:", st.session_state.count)

        # with nth count go to nth gapped_passage entry
        st.session_state.prompt = testing_message[st.session_state.count]
        st.write("sessionstate prompt is:", st.session_state.prompt)
        assemble_pre_prompt(0)

        # generate text for nth passage entry
        if not st.session_state.has_finished:
            stream_response(0)
        else:
            # prepare new cycle
            st.session_state.user_msgs.append(testing_message[st.session_state.count])  # static
            st.session_state.assistant_msgs[0].append(st.session_state.response[0])

        st.write(st.session_state.response[0][-1])
        # append everything


        if f'response{0}' in st.session_state:
            st.session_state.assistant_msgs[0].append(st.query_params[f"response{0}"])

    if reset_count:
        st.session_state.count = 0




    # for i in range(st.session_state.amount_of_inputs):  # corresponds to first up to previous
    #     with st.chat_message(name="user", avatar="user"):
    #         st.write(st.session_state.all_user_messages[i])
    #     for j in range(st.session_state.amount_of_responses):
    #         with st.chat_message(name="assistant", avatar="assistant"):
    #             st.write(st.session_state.all_assistant_messages[j][i])
    #
    # if st.session_state.prompt:
    #     with st.chat_message(name="assistant", avatar="user"):
    #         st.write(st.session_state.prompt)
    #     for i in range(st.session_state.amount_of_responses):
    #         with st.chat_message(name="assistant", avatar="assistant"):
    #             st.write(stream_response(i))
    #         print("appending:", st.session_state.response[i])
    #         st.session_state.all_assistant_messages[i].append(st.session_state.response[i])  # increase size by one and fill with response
    #
    #         st.session_state.something_downloadable = True
    #     st.session_state.all_user_messages.append(st.session_state.prompt)
    #     st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]  # empty responses
    #     st.session_state.amount_of_inputs += 1
    #
    # if st.session_state.something_downloadable:
    #     st.download_button("download responses", "\n\n".join([assemble_pre_prompt(i) for i in range(st.session_state.amount_of_responses)]))


# Run the main function

main()
