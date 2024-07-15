import json
import streamlit as st

import torch
from streamlit.errors import StreamlitAPIException

from cuda_stuffs import update_cuda_stats_at_progressbar

predefined_questions = {
    1: "Please restore the given text fragment",
    2: "Do you know about this text?",
    3: "What genre is this text about?",
    4: "Who is the author of this text?"

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
    to_be_saved_things = ["assistant_msgs", "user_msgs", "ground_truth", "amount_of_to_be_processed_paragraphs", "system"]
    state_data = {key: value for key, value in st.session_state.items() if not key.startswith('_') and key in to_be_saved_things}
    json_data = json.dumps(state_data)
    st.download_button(
        label="Download state",
        file_name="sessionstate/session_state.json",
        mime="application/json",
        data=json_data,
        use_container_width=True
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


def disable_output():
    if (len(st.session_state.user_msgs)) * st.session_state.repeat_count_per_paragraph > 7:
        st.session_state.disabled = True
    else:
        st.session_state.disabled = False


def create_sidebar():
    with st.sidebar:
        st.logo('logo.svg')

        st.markdown(":gray[**Reduce the amount of text covering the screen**]")
        st.session_state.show_system_prompt = False
        st.session_state.show_user_message = False
        st.session_state.show_assistant_message = False
        st.session_state.repeat_count_per_paragraph = st.number_input("repeat amount for current paragraph", step=1, value=1, min_value=1, max_value=50, on_change=disable_output())
        st.session_state.dataset = st.text_area(label="Dataset input",
                                                value="After experiencing previous OCR image recognition, a series of actionable text data will be obtained, which contains useful information required for the task, as well as a large number of irrelevant characters such as spaces and line breaks. Therefore, it is necessary to preprocess the text data first, including removing spaces, line breaks, and formatting specific formatted data. In the preprocessed data, the position information of relevant fields will also be retained, which refers to the relative position and layout rules of the fields in the form. This can help us to better understand the structure and context of the form. For tasks with predefined form templates, this can greatly simplify the difficulty of field matching. We can directly narrow the matching range based on the predefined field position, and for predefined forms, their corresponding fields are also very similar, so field matching can be performed directly on a small scale based on the field names defined in the form template, and filtering can be carried out using simple regular expressions or string matching. However, this approach also has its limitations. For cases where the form framework is not fixed, the usefulness of field position information may not be as obvious, as the field positions may vary greatly across different forms. In addition, this method cannot accurately fill in some fields with different names but similar meanings.")

        st.write("Predefined question(s)")
        st.text_area("system - *let the LLM know about specific information*", key="s1", value="Please fill in the missing characters in the given text fragment from a 2024 scientific literature on OCR and text preprocessing, ensuring that the restoration maintains the original language style, scientific accuracy, and logical flow. In the following text, each dash character represents a missing word. Your task is to follow the instructions to restore the missing text. Please note the response format: output only the complete, restored text itself without adding any additional narrative.")

        # number input -> amount of questions with key = "q"+i -> collect questions afterward

        amount_of_questions = st.number_input("amount of questions", step=1, value=1, min_value=1, max_value=50, on_change=disable_output())
        for i in range(1, amount_of_questions + 1):
            try:
                st.text_area(f"question {i}", key=f"q{i}", value=predefined_questions[i])
            except KeyError:
                st.text_area(f"question {i}", key=f"q{i}", value="Improve your text further!")

        # predefine user input OwO
        sorted_keys = sorted((key for key in st.session_state if key.startswith("q")), key=lambda x: int(x[1:]))
        st.session_state.user_msgs = [st.session_state[key] for key in sorted_keys]
        st.session_state.response = ""

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
                clear_cache = st.button("clear cache")
                if clear_cache:
                    st.cache_resource.clear()
                st.session_state.save_whole_state = st.empty()
                to_be_saved = ["system", "ground_truth", "gapped_results", "assistant_msgs", "user_msgs", "BLEU", "BERT", "METEOR"]
                settings_to_download = {k: v for k,v in st.session_state.items() if k in to_be_saved}
                st.session_state.save_whole_state = st.download_button(label="Download whole state",
                                                       data=json.dumps(settings_to_download),
                                                       file_name=f"settings.json",
                                                       help="Click to Download Current Settings")

        if torch.cuda.is_available():
            st.info(f"CUDA is available")
        else:
            st.warning("CUDA is currently not available - system might be out of VRAM")
            st.toast("CUDA is currently not available - system might be out of VRAM")
        st.markdown(
            """
            <style>
            .stButton > button {
                width: 100%;
                margin-top: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.vram_empty = st.empty()
        if st.button("empty CUDA cache"):
            torch.cuda.empty_cache()
            update_cuda_stats_at_progressbar()


def create_main_buttons():
    return st.button("Start computation", on_click=disable_output())


def create_progress_bars():
    paragraph_progress = st.progress(0, text="processing paragraph")
    question_progress = st.progress(0, text="processing questions")
    time_placeholder = st.empty()
    estimate_placeholder = st.empty()
    return paragraph_progress, question_progress, time_placeholder, estimate_placeholder
