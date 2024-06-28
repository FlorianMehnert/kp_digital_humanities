from dataclasses import dataclass

from pages.LLM.Prompt import Prompt
import streamlit as st


@dataclass
class Controller:
    def __init__(self, instances: int, temperature: float, top_p: float, system):
        self.instances = instances
        self.temperature = temperature
        self.top_p = top_p
        self.prompts: list[Prompt] = [Prompt(system) for _ in range(instances)]
        self.system = system

    def change_system(self, new_system_msg):
        for p in self.prompts:
            p.system = new_system_msg

    def on_change_amount_responses(self):
        """
        either this one adds more responses based on current instance and tries to keep up or
        empties everything up to current number of instances.
        """
        if len(self.prompts) > self.instances:
            self.prompts = self.prompts[:self.instances]
        else:
            additional: int = self.instances - len(self.prompts)
            new_prompts = [Prompt(self.system) for _ in range(additional)]
            self.prompts = self.prompts + new_prompts
            #st.write([a.user for a in self.prompts], [a.assistant for a in self.prompts])

            # TODO: generate based on current steps? ofc hehe

    def add_user_input(self, user_msg: str):
        for prompt in self.prompts:
            prompt.user.append(user_msg)

    def pre_render(self, st):
        for prompt in self.prompts:
            for i in range(len(prompt.assistant)):
                with st.chat_message(name="user", avatar="user"):
                    st.write(prompt.user[i])
                with st.chat_message(name="assistant", avatar="assistant"):
                    st.write(prompt.assistant[i])

    def add_answer(self, i, answer):
        self.prompts[i].user.append(answer)

    def reset_everything(self):
        for prompt in self.prompts:
            prompt.assistant = []
            prompt.user = []
