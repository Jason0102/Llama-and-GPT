import requests
import numpy as np
import json
import os

# O1_PREVIEW_URL = 'https://api.openai.com/v1/engines/o1-preview/completions'
# O1_MINI_URL = 'https://api.openai.com/v1/engines/o1-mini/completions'
CHAT_URL = 'https://api.openai.com/v1/chat/completions'
EMBEDDING_URL = 'https://api.openai.com/v1/embeddings'

class GPT():
    def __init__(self, openai_api_key:str, prompt, temperature = 0, model="gpt-3.5-turbo", text_memory=None, img_memory=None) -> None:
        self.key = openai_api_key
        self.prompt = prompt
        self.text_stm = text_memory
        self.img_stm = img_memory
        self.temperature = temperature
        self.model = model

    def run(self, text_dict: dict, img_list=[]) -> str:
        send = []
        # load img
        if self.img_stm != None:
            if img_list != []:
                self.img_stm.refresh()
                for img in img_list:
                    send.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                            "detail": "low"
                    }})
            else: 
                for img in self.img_stm.get_img():
                    send.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                            "detail": "low"
                        }})
        if img_list != None:
            for img in img_list:
                send.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}",
                        "detail": "low"
                }})
                    
        # load text 
        if self.text_stm != None:
            chat_history = self.text_stm.get()
            text_dict['chat_history'] = chat_history

        text = self.prompt.format(text_dict)
        send.append({
            "type": "text",
            "text": text
        })

        # form request
        message = [{
            "role": "user",
            "content": send
            }]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
            }
        if self.model == 'o1-preview' or self.model == 'o1-mini':
            payload = {
                "model": self.model,
                "messages":  message,
                }  
        else:    
            payload = {
                "model": self.model,
                "messages":  message,
                "temperature": self.temperature,
                "max_tokens": 1024,
            }

        for i in range(5):
            try:
                response = requests.post(CHAT_URL, headers=headers, json=payload)
                j = response.json()

                return j['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                print(j["error"]["message"])
                continue
        return 'gpt error'
    
class Embedding():
    def __init__(self, openai_api_key:str, folder:str, model:str) -> None:
        self.folder = folder
        self.key = openai_api_key
        self.model = model
        self.fileList = []
        self.documents = []
        self.vector_store = []


    def load_docs(self) -> list:
        print('Loading documents......')
        docs = []
        if self.fileList == []:
            self.fileList = os.listdir(self.folder)
        for file in self.fileList:
            if file[-4:] == '.txt':
                path = self.folder + file
                with open(path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    docs.append(text)
            else:
                del file
        return docs
    
    def get_embedding(self, text:str):
        headers = {
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json'
        }
        data = {
            "input": text,
            "model": self.model
        }
        for i in range(5):
            try:
                response = requests.post(EMBEDDING_URL, headers=headers, json=data)
                j = response.json()
                return j['data'][0]['embedding']    
            except Exception as e:
                print(e)
                print(j["error"]["message"])
                continue
        return 'embedding error'
    
    def mmr(self, doc_embeddings, query_embedding, lambda_param=0.5):

        query_similarities = np.array([np.dot(query_embedding, doc_embedding) for doc_embedding in doc_embeddings])
        
        selected_docs = [doc_embeddings[np.argmax(query_similarities)]]
        doc_embeddings = np.delete(doc_embeddings, np.argmax(query_similarities), axis=0)
        query_similarities = np.delete(query_similarities, np.argmax(query_similarities))
        
        while len(doc_embeddings) > 0:
            mmr_scores = lambda_param * query_similarities - (1 - lambda_param) * np.max(
                [np.dot(doc_embeddings, selected_doc) for selected_doc in selected_docs])
            best_doc_index = np.argmax(mmr_scores)
            selected_docs.append(doc_embeddings[best_doc_index])
            
            doc_embeddings = np.delete(doc_embeddings, best_doc_index, axis=0)
            query_similarities = np.delete(query_similarities, best_doc_index)
        
        return selected_docs

    def build_db(self):
        self.documents = self.load_docs()
        print('Building vector database......')
        for doc in self.documents:
            self.vector_store.append(self.get_embedding(doc))
        print(f'Vector database size {len(self.vector_store)}')

    def load_db(self, path:str) -> int:
        if not os.path.isfile(path):
            print('file not found')
            return -1
        if path.find('.json') == -1:
            print('not .json')
            return -1
        with open(path, 'r') as json_file:
            loaded_data = json.load(json_file)
        self.fileList = loaded_data[0]
        self.documents = self.load_docs()
        self.vector_store = loaded_data[1]
        print(f'Vector database {path} loaded, size: {len(self.vector_store)}')
        return 0
    
    def save_db(self, path:str) -> int:
        with open(path, 'w') as json_file:
            json.dump([self.fileList, self.vector_store], json_file)
        print(f'Vector database {path} saved')
        return 0

    def retrieve(self, query:str, k=1, method='mmr') -> str:
        if k < 1:
            return 'error of k'
        text = ''
        if method == 'mmr':
            query_embedding = self.get_embedding(query)
            sort_embeddings = self.mmr(self.vector_store, query_embedding) 
            sorted_docs = [self.documents[np.argmax([np.dot(sort_embedding, vector) for vector in self.vector_store])] for sort_embedding in sort_embeddings]
            if k > len(sorted_docs):
                k = len(sorted_docs)
            for i in range(k):
                text = text + sorted_docs[i] + '\n'
            
        return text
    
    def add_doc(self, file_name:str) -> int:
        path = self.folder + file_name
        with open(path, 'r', encoding='utf-8') as file:
            new_text = file.read()
        new_vector = self.get_embedding(new_text)
        self.fileList.append(file_name)
        self.vector_store.append(new_vector)
        self.documents.append(new_text)
        return 0
    
    def remove_doc(self, file_name:str) -> int:
        i = 0
        for file in self.fileList:
            if file == file_name:
                del self.fileList[i]
                del self.documents[i]
                del self.vector_store[i]
                print('file removed from vector db')
                return 0
            i = i + 1
        print(f'filename: {file_name} not in the vector db')
        return -1





    
