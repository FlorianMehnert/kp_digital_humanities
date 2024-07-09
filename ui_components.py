import json
import streamlit as st
import random

from streamlit.errors import StreamlitAPIException

predefined_questions = {
    1: "In the provided text missing words are marked with a minus sign. Insert the missing words. Only respond with the corrected text. Do not add any summarization.",
    2: "Improve your text further!",
    3: "Improve your text further!"
}

debug_namings = {
    "assistant_msgs": ("paragraph", "question"),
    "user_msgs": ("question", ""),
    "ground_truth": ("paragraph", "")
}

debug_input_visibility = {
    "assistant_msgs": (False, False),
    "user_msgs": (False, True),
    "ground_truth": (False, True)
}


def save_state():
    to_be_saved_things = ["assistant_msgs", "user_msgs", "ground_truth_msgs", "amount_of_to_be_processed_paragraphs", "system"]
    state_data = {key: value for key, value in st.session_state.items() if not key.startswith('_') and key in to_be_saved_things}
    json_data = json.dumps(state_data)
    st.download_button(
        label="Download state",
        file_name="session_state.json",
        mime="application/json",
        data=json_data
    )


# Function to load the session state
def load_state(file):
    if file is not None:
        try:
            state_data = json.load(file)
            for key, value in state_data.items():
                st.session_state[key] = value
            st.success("Session state loaded successfully!")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid session state file.")
        except StreamlitAPIException:
            pass


def create_sidebar():
    with st.sidebar:
        st.logo('logo.svg')

        st.markdown(":gray[**Reduce the amount of text covering the screen**]")
        col1, col2, col3 = st.columns(3)
        lag_decrease = True if (len(st.session_state.user_msgs)-1) * st.session_state.repeat_count_per_paragraph < 8 else False
        with col1:
            st.session_state.show_system_prompt = st.toggle("system", value=lag_decrease, disabled=not lag_decrease)
        with col2:
            st.session_state.show_user_message = st.toggle("user", value=lag_decrease, disabled=not lag_decrease)
        with col3:
            st.session_state.show_assistant_message = st.toggle("assistant", value=lag_decrease, disabled=not lag_decrease)

        with st.expander("**Predefined questions**"):
            st.text_area("system 1", key="s1", value="The following text is missing one or multiple words. Your task is to listen to the following tasks. ")

            # number input -> amount of questions with key = "q"+i -> collect questions afterward

            amount_of_questions = st.number_input("amount of questions", step=1, value=1, min_value=1, max_value=50)
            for i in range(1, amount_of_questions + 1):
                try:
                    st.text_area(f"question {i}", key=f"q{i}", value=predefined_questions[i])
                except KeyError:
                    st.text_area(f"question {i}", key=f"q{i}", value="Improve your text further!")

        # predefine user input OwO
        sorted_keys = sorted((key for key in st.session_state if key.startswith("q")), key=lambda x: int(x[1:]))
        st.session_state.user_msgs = [st.session_state[key] for key in sorted_keys]
        st.session_state.response = ""

        with st.expander("**Load/Save Session**"):
            if st.button("Save State"):
                save_state()

            # Load state file uploader
            uploaded_file = st.file_uploader("Load State", type="json")
            if uploaded_file is not None:
                if st.button("Load State"):
                    load_state(uploaded_file)

        advanced_settings = st.checkbox("Advanced Settings")
        if advanced_settings:
            with st.expander("**LLM Parameters**"):
                st.session_state.temperature = st.slider("**Temperature:** by default 0.97 but adjust to your needs:", min_value=0.0, value=0.97, max_value=10.0)
                st.session_state.num_predict = st.slider("**Max tokens**: Maximum amount of tokens that are output:", min_value=128, value=512, max_value=2048)
                st.session_state.top_p = st.slider("**Top p**: By default 0.9 - lower top p means llama will select more unlikely tokens more often", min_value=0.0, value=0.9, max_value=1.0)
            with st.expander("**Obfuscation Parameters**"):
                st.session_state.mask_rate = st.slider("mask rate", 0.0, 1.0, 0.3)
                st.session_state.seed = st.slider("seed", 0, 128, 69)
                random.seed(st.session_state.seed)
            with st.expander("**Debug**"):
                col1, col2 = st.columns(2)
                option = st.selectbox("Which data do you want to display?", ("assistant_msgs", "user_msgs", "ground_truth", "gapped_results"))
                depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1
                try:
                    with col1:
                        column = st.number_input(label=debug_namings[option][0], step=1, min_value=0, disabled=debug_input_visibility[option][0])
                    with col2:
                        row = st.number_input(label=debug_namings[option][1], step=1, min_value=0, disabled=debug_input_visibility[option][1])
                    if depth(st.session_state[option]) == 1:
                        st.info("only column influences data")
                        st.write(st.session_state[option][int(column)])
                    elif depth(st.session_state[option]) == 2:
                        st.write(st.session_state[option][int(column)][int(row)])
                except ValueError:
                    st.warning("Generate some text first")
                except IndexError:
                    st.warning("Index Error")


def create_main_buttons():
    col1, col2, col3 = st.columns(3)
    with col1:
        start_computation = st.button("Start computation")
    with col2:
        repeat_count = st.number_input("repeat amount for current paragraph", step=1, value=1, min_value=1, max_value=50)
    with col3:
        plot_diagram = st.toggle("Plot diagram")
    return start_computation, repeat_count, plot_diagram


def create_progress_bars():
    paragraph_progress = st.progress(0, text="processing paragraph")
    question_progress = st.progress(0, text="processing questions")
    time_placeholder = st.empty()
    estimate_placeholder = st.empty()
    return paragraph_progress, question_progress, time_placeholder, estimate_placeholder