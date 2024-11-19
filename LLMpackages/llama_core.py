from llama_cpp import Llama

class llama_core():
    def __init__(self, model:str, is_GPU=False, ):
        if is_GPU:
            self.llm = Llama(
                model_path=model, 
                n_gpu_layers=-1, 
                verbose = False,
            )
        else:
            self.llm = Llama(
                model_path=model, 
                verbose = False,
            )
        output = self.llm(
            "Q: Name the planets in the solar system? A: ", 
            max_tokens=128, 
            stop=["Q:"], 
            echo=False, 
            temperature=0,
            stream=True
        ) 
        self.agents = []
        self.id_list = []
        print("Start running Llama .........")

    def create_llama_agent(self, prompt_template, id:int, buffer=None, temperature=0.2) -> int:
        if temperature < 0 or temperature > 1:
            raise TypeError("Invalid temperature")
        for agent in self.agents:
            if id == agent['id']:
                agent['prompt_template'] = prompt_template
                agent['temperature'] = temperature
                agent['buffer'] = buffer
                print(f"Llama agent id={id} updated")
                return 0
        self.agents.append({"id": id, "prompt_template": prompt_template, "buffer": buffer, "temperature": temperature})
        print(f"Llama agent id={id} created")
        return 0

    def agent_run(self, input_dict:dict, id:int) -> str:
        for agent in self.agents:
            if id == agent['id']:
                if agent['buffer'] != None:
                    input_dict["chat_history"] = agent['buffer'].get()
                prompt = agent['prompt_template'].format(input_dict)
                print(prompt)
                output = self.llm(
                    prompt, 
                    max_tokens=1024, 
                    # stop=["\n"], 
                    echo=False, 
                    temperature=agent['temperature'],
                ) 
                return str(output["choices"][0]['text'])
            else:
                continue
        raise TypeError("Agent id not found")
