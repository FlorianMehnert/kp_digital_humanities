import streamlit as st
from ollama import generate
from enum import Enum

system = "You are an assistant. You try to find characters that do not belong in the given sentence. Only respond with the corrected sentence. Do not add any summarization."

# Initialize session state variables
if 'has_finished' not in st.session_state:
    st.session_state.has_finished = False
if 'amount_of_responses' not in st.session_state:
    st.session_state.amount_of_responses = 3
if 'response' not in st.session_state:
    st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]  # current responses
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


class Roles(Enum):
    system: str = f"{start_token}system{end_token_role}\n"
    assistant: str = f"{start_token}assistant{end_token_role}\n"
    user: str = f"{start_token}user{end_token_role}\n"


# available per instance
class Prompt:
    def __init__(self, system):
        self.begin_token = "<|begin_of_text|>"
        self.start_token = "<|start_header_id|>"
        self.end_token_role = "<|end_header_id|>"
        self.end_token_input = "<|eot_id|>"
        self.system = system
        self.user: list[str] = []
        self.assistant: list[str] = []

    def system_prompt(self):
        return f'{begin_token}{Roles.system}{self.system}{end_token_input}{Roles.user}'

    def assemble_pre_prompt(self):
        prompt: str = self.system_prompt()
        for i in range(st.session_state.amount_of_inputs):
            prompt += self.user[i]
            prompt += end_token_input
            prompt += str(Roles.assistant)  # corresponding to the next message type

            prompt += self.assistant[i]
            prompt += end_token_input
            prompt += str(Roles.user)
        return prompt

    def add_user_input(self, prompt):
        self.user.append(prompt)

    def stream_response(self, idx):
        response = generate(
            model='llama3:instruct',
            # prompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{st.session_state.system}<|eot_id|><|start_header_id|>user<|end_header_id|>{st.session_state.prompt}<|start_header_id|>assistant<|end_header_id|>',
            prompt=self.assemble_pre_prompt() + st.session_state.prompt + end_token_input + str(Roles.assistant),
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
            #st.session_state.response[idx] += chunk.get("response")
            yield chunk.get("response")
        self.assistant.append(st.session_state.response[idx])


class Controller:
    def __init__(self, instances: int, temperature: float, top_p: float):
        self.instances = instances
        self.temperature = temperature
        self.top_p = top_p
        self.prompts: list[Prompt] = [Prompt(st.session_state.system) for _ in range(st.session_state.amount_of_inputs)]

    def change_system(self, new_system_msg):
        for p in self.prompts:
            p.system = new_system_msg

    def on_change_amount_responses(self):
        """
        either this one adds more responses based on current instance and tries to keep up or
        empties everything up to current number of instances.
        """
        print("in change amount")

        if len(self.prompts) > self.instances:
            self.prompts = self.prompts[:self.instances]
        else:
            additional: int = self.instances - len(self.prompts)
            new_prompts = [Prompt(st.session_state.system) for _ in range(additional)]
            self.prompts = self.prompts + new_prompts
            st.write([a.user for a in self.prompts], [a.assistant for a in self.prompts])
            # TODO: generate based on current steps? ofc hehe

    def add_user_input(self, user_msg: str):
        for prompt in self.prompts:
            prompt.add_user_input(user_msg)

    def pre_render(self):
        for prompt in self.prompts:
            for i in range(len(prompt.assistant)):
                with st.chat_message(name="user", avatar="user"):
                    st.write(prompt.user[i])
                with st.chat_message(name="assistant", avatar="assistant"):
                    st.write(prompt.assistant[i])

    def add_answer(self, i, answer):
        self.prompts[i].assistant.append(answer)

    def reset_everything(self):
        for prompt in self.prompts:
            prompt.assistant = []
            prompt.user = []


if 'overlord' not in st.session_state:
    st.session_state.overlord = Controller(st.session_state.amount_of_responses, st.session_state.temperature, st.session_state.top_p)


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
    st.title(f"Llama {"".join([":llama:" for _ in range(3)])} playground")
    sidebar()
    st.write([(a.assistant, a.user) for a in st.session_state.overlord.prompts])

    st.session_state.overlord.pre_render()
    new_input = st.chat_input(placeholder="Your message", key="prompt_input", disabled=False)
    if new_input:

        # display prompt
        with st.chat_message(name="assistant", avatar="user"):
            st.write(new_input)

        # generate answer
        st.write("prompts length is:", len(st.session_state.overlord.prompts))
        for i in range(len(st.session_state.overlord.prompts)):
            with st.chat_message(name="assistant", avatar="assistant"):
                st.write(st.session_state.overlord.prompts[i].stream_response(i))
                st.write(st.session_state.overlord.prompts[i].stream_response(i))
        st.session_state.overlord.add_answer(i, st.session_state.response[i])

        st.session_state.overlord.add_user_input(new_input)
        st.session_state.response = ["" for _ in range(st.session_state.amount_of_responses)]  # empty responses


main()
