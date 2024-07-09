from enum import Enum

import streamlit as st
from ollama import generate

# construct llama3 prompts
begin_token = "<|begin_of_text|>"
start_token = "<|start_header_id|>"
end_token_role = "<|end_header_id|>"
end_token_input = "<|eot_id|>"


class Roles(Enum):
    system: str = f"{start_token}system{end_token_role}\n"
    assistant: str = f"{start_token}assistant{end_token_role}\n"
    user: str = f"{start_token}user{end_token_role}\n"


def system_prompt() -> str:
    return f'{begin_token}{Roles.system.value}{st.session_state.system}{end_token_input}{Roles.user.value}'


def assemble_pre_prompt(idx: int) -> str:
    """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant for travel tips and recommendations<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    What is France's capital?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>
    What can I do there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Paris, the City of Light, offers a romantic getaway with must-see attractions like the Eiffel Tower and Louvre Museum, romantic experiences like river cruises and charming neighborhoods, and delicious food and drink options, with helpful tips for making the most of your trip.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Give me a detailed list of the attractions I should visit, and time it takes in each one, to plan my trip accordingly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt: str = system_prompt()
    prompt += st.session_state.user_msgs[0]
    prompt += end_token_input
    prompt += str(Roles.assistant.value)

    for i in range(st.session_state.count + 1):  # count usually starts with 0 -> range(0) is nothing
        prompt += st.session_state.assistant_msgs[idx][i] if st.session_state.assistant_msgs else ""  # avoiding potential exception
        prompt += end_token_input
        prompt += str(Roles.user.value)

        prompt += st.session_state.user_msgs[i]
        prompt += end_token_input
        prompt += str(Roles.assistant.value)
    return prompt


def stream_response(idx: int, stream=True):
    """
    generating response for the nth assistant where n is idx
    using st.assistant_msgs and st.user_msgs for preprompt
    """
    response = generate(
        model='llama3:instruct',
        prompt=assemble_pre_prompt(idx),
        options={
            'num_predict': st.session_state.num_predict,
            'temperature': st.session_state.temperature,
            'top_p': st.session_state.top_p,
            'stop': ['<EOT>'],
        },
        stream=stream
    )
    if stream:
        # yield response
        for chunk in response:
            st.session_state.response += chunk.get("response")
            yield chunk.get("response")
            if chunk.get("done"):
                st.session_state.has_finished = True
                if chunk.get("done_reason") == "length":
                    st.warning("Please increase the LLM Parameter Max tokens")
    else:
        st.session_state.response = response.get("response")
        st.session_state.has_finished = True
        return response.get("response")


def process_llm_responses(paragraph_repetition, user_msgs, assistant_msgs, paragraph):
    st.session_state.system = st.session_state.s1 + "**" + paragraph + "**\n"

    if st.session_state.show_system_prompt:
        with st.chat_message("system", avatar="./images/system.svg"):
            st.subheader("system prompt")
            st.write(st.session_state.system)

    st.session_state.count = -1

    for question_number in range(len(user_msgs)):
        if st.session_state.has_finished:
            st.session_state.count += 1

            try:
                user_msgs[st.session_state.count]
            except IndexError:
                user_msgs.append("")

            for assistant in assistant_msgs:
                try:
                    assistant[st.session_state.count]
                except IndexError:
                    assistant.append("")

            st.session_state.has_finished = False

        if not st.session_state.has_finished:
            if st.session_state.show_user_message:
                with st.chat_message("user"):
                    st.write(user_msgs[st.session_state.count])
            if st.session_state.show_assistant_message:
                with st.chat_message("assistant"):
                    st.write(stream_response(paragraph_repetition))
            else:
                st.write(stream_response(paragraph_repetition, False))

        if st.session_state.has_finished:
            assistant_msgs[paragraph_repetition][st.session_state.count] = st.session_state.response
            st.session_state.response = ""
