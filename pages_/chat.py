import streamlit as st
from pages_.LLM.Controler import Controller

system = "You are an assistant. You try to find characters that do not belong in the given sentence. Only respond with the corrected sentence. Do not add any summarization."

# Initialize session state variables
if 'has_finished' not in st.session_state:
    st.session_state.has_finished = False
if 'amount_of_responses' not in st.session_state:
    st.session_state.amount_of_responses = 3
if 'response' not in st.session_state:
    st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]  # current responses
if 'all_user_messages' not in st.session_state:
    st.session_state.user_msgs = []  # [['' for _ in range(1)] for _ in range(st.session_state.amount_responses)]
if 'all_assistant_messages' not in st.session_state:
    st.session_state.assistant_msgs = [[], [], []]  # [['' for _ in range(1)] for _ in range(st.session_state.amount_responses)]
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

if 'overlord' not in st.session_state:
    st.session_state.overlord = Controller(st.session_state.amount_of_responses, st.session_state.temperature, st.session_state.top_p, system=system)


def sidebar():
    with st.sidebar:
        st.session_state.system = st.text_area("System prompt (e.g. preprompt - instructions)", key="system_input", value=system)
        st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:", min_value=0.0, value=0.97, max_value=10.0)
        st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128, value=128, max_value=2048)
        st.session_state.top_p = st.slider("**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often", min_value=0.0, value=0.9, max_value=1.0)
        st.session_state.amount_of_responses = st.slider("amount of responses", min_value=1, value=3, max_value=50, key="response_slider", disabled=False)
        st.session_state.overlord.instances = st.session_state.amount_of_responses
        st.session_state.overlord.on_change_amount_responses()
        st.session_state.disallow_multi_conversations = st.button("reset converations", on_click=st.session_state.overlord.reset_everything())  # reset chatter input


def main():
    st.logo("https://ollama.com/public/ollama.png")
    type(st)
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")
    sidebar()

    st.session_state.overlord.pre_render(st)
    new_input = st.chat_input(placeholder="Your message", key="prompt_input", disabled=False)
    if new_input:
        for prompt in st.session_state.overlord.prompts:
            prompt.user.append(new_input)
        # display prompt
        with st.chat_message(name="assistant", avatar="user"):
            st.write(new_input)

        # generate answer per instance
        for i in range(len(st.session_state.overlord.prompts)):

            # write as separate chat message
            with st.chat_message(name="assistant", avatar="assistant"):
                chunk = st.session_state.overlord.prompts[i].stream_response(i, new_input)
                st.write(chunk)
        print("user list:", st.session_state.overlord.prompts[-1].user)
        print("assistant list:", st.session_state.overlord.prompts[-1].assistant)
        st.session_state.overlord.add_user_input(new_input)


main()
