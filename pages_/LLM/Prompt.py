from dataclasses import dataclass

from ollama import generate
from pages_.LLM.Roles import Roles

# construct llama3 prompts
begin_token = "<|begin_of_text|>"
start_token = "<|start_header_id|>"
end_token_role = "<|end_header_id|>"
end_token_input = "<|eot_id|>"


@dataclass
class Prompt:
    def __init__(self, system):
        self.begin_token = "<|begin_of_text|>"
        self.start_token = "<|start_header_id|>"
        self.end_token_role = "<|end_header_id|>"
        self.end_token_input = "<|eot_id|>"
        self.system = system
        self.user: list[str] = []
        self.assistant: list[str] = []
        self.predict = 128
        self.temperature = 0.97
        self.top_p = 0.9

    def system_prompt(self):
        return f'{begin_token}{Roles.system}{self.system}{end_token_input}{Roles.user}'

    def assemble_pre_prompt(self):
        prompt: str = self.system_prompt()
        print("len ass", len(self.assistant))
        print("len user", len(self.user))
        print(self.assistant[0])
        for i in range(len(self.assistant)):
            prompt += self.user[i]
            prompt += end_token_input
            prompt += str(Roles.assistant)  # corresponding to the next message type

            prompt += self.assistant[i]
            prompt += end_token_input
            prompt += str(Roles.user)
        return prompt

    def add_user_input(self, prompt):
        self.user.append(prompt)

    def stream_response(self, idx, prompt):
        self.assistant.append("")
        response = generate(
            model='llama3:instruct',
            # prompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{st.session_state.system}<|eot_id|><|start_header_id|>user<|end_header_id|>{st.session_state.prompt}<|start_header_id|>assistant<|end_header_id|>',
            prompt=self.assemble_pre_prompt() + prompt + end_token_input + str(Roles.assistant),
            options={
                'num_predict': self.predict,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'stop': ['<EOT>'],
            },
            stream=True
        )
        # yield response

        for chunk in response:
            self.assistant[-1] += chunk.get("response")
            yield chunk.get("response")
