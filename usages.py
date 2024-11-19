from LLMpackages.llama_core import llama_core
from LLMpackages.LLMopenai import GPT, Embedding
from LLMpackages.PromptTemplate import PromptTemplate
from LLMpackages.Buffer import TextBuffer, encode_image
from datetime import datetime

API_KEY='your openai key'

def timelabel():
    currentDateAndTime = datetime.now()
    currentDateString = str(currentDateAndTime.year) + '-' + str(currentDateAndTime.month) + '-' + str(currentDateAndTime.day) + '-' + currentDateAndTime.strftime("%H-%M-%S")
    return currentDateString

## Usages of GPT in API
def gpt_example():
    prompt = PromptTemplate("./prompts/prompt.txt")
    agent = GPT(openai_api_key=API_KEY, prompt=prompt, model="gpt-4o-mini",temperature=0.2)
    print(timelabel())
    output = agent.run({'text': '當台北的朋友跟你說下次約，你要怎麼做?從以下選項擇一: 1. 馬上確認具體時間 2. 不主動詢問但等待對方來約 3. 當作沒這回是，一切隨緣'})
    print(output)
    print(timelabel())

def gpt_chat():
    prompt = PromptTemplate("./prompts/prompt.txt")
    buffer = TextBuffer(buffer_size=3)
    agent = GPT(openai_api_key=API_KEY, prompt=prompt, model="gpt-4o-mini", text_memory=buffer, temperature=0.2)
    while True:
        text = input('請輸入: ')
        print(timelabel())
        input_dict = {'text': text}
        output = agent.run(input_dict)
        print(output)
        print(timelabel())
        buffer.set({"input": input_dict, "output": output})

def gpt_img():
    prompt = PromptTemplate("./prompts/prompt.txt")
    buffer = TextBuffer(buffer_size=3)
    agent = GPT(openai_api_key=API_KEY, prompt=prompt, model="gpt-4o-mini", text_memory=buffer, temperature=0.2)
    text_dict = {'text': '你看到了幾個杯子?他們是什麼顏色的?'}
    path = './input_pictures/IMG_8405.jpg'
    img_list = [encode_image(path)]
    result = agent.run(text_dict, img_list)
    print(result)

## Usages of Llama in local
def llama_example():
    core = llama_core(model="./models/Taiwan-LLM-13B-v2.0-chat-Q8_0.gguf", is_GPU=True)
    prompt_template = PromptTemplate("./prompts/prompt.txt")
    core.create_llama_agent(prompt_template=prompt_template, id=0)
    input_dict = {"text": "請用200字介紹台灣八加九文化"}
    print(timelabel())
    output = core.agent_run(input_dict=input_dict, id=0)
    print(output)
    print(timelabel())

def chat_llama():
    core = llama_core(model="./models/Taiwan-LLM-13B-v2.0-chat-Q8_0.gguf", is_GPU=True)
    prompt_template = PromptTemplate("./prompts/prompt.txt")
    buffer = TextBuffer(buffer_size=3)
    core.create_llama_agent(prompt_template, 0, buffer=buffer, temperature=0.8)
    while True:
        text = input('請輸入: ')
        print(timelabel())
        input_dict = {'text': text}
        output = core.agent_run(input_dict, 0)
        print(output)
        print(timelabel())
        buffer.set({"user": input_dict, "agent": output})

## Usages of OpenAI embedding

DIR='folder of documents'

def construct_vector_db(): # 建立並儲存向量資料庫
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR,
        model=MODEL
    )
    model.build_db()
    model.save_db('text_db.json')

def load_and_retrieval():# 載入並提取相關資料
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR,
        model="text-embedding-3-large"
    )
    model.load_db('text_db.json') 
    query = '聖誕節'
    text = model.retrieve(query, k=3)
    print(text)

def load_and_add_file():# 載入並添加單一文件至向量資料庫
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR,
        model="text-embedding-3-large"
    )
    model.load_db('text_db.json') 
    model.add_doc('test.txt') # file test.txt must in folder DIR
    model.save_db('text_db.json')

def load_and_remove_file():# 載入並從向量資料庫移除單一文件
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR,
        model="text-embedding-3-large"
    )
    model.load_db('text_db.json') 
    model.remove_doc('test.txt') # file test.txt must in folder DIR