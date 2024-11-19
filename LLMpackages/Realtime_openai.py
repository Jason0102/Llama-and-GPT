import websockets
import asyncio
import json
import threading
# import io
import json
# from pydub import AudioSegment
import base64
import speech_recognition as sr
import pyaudio
import wave

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
FORMAT = pyaudio.paInt16  # 16位元格式
CHANNELS = 1  # 單聲道
RATE = 44100  # 取樣率
CHUNK = 1024  # 每次讀取的幀數
RECORD_SECONDS = 3  # 錄音時間
WAVE_OUTPUT_FILENAME = "output.wav"


class Realtime_openai():
    def __init__(self, key:str, prompt=None, text_memory=None, mode='audio') -> None:
        self.input_dict = {}
        self.text_stm = text_memory
        self.mode = mode
        if prompt != None:
            self.prompt = prompt
        else:
            self.prompt = ''
        self.audio_output = []
        self.text_output = []
        self.full_text_output = ''
        self.headers = {
            "Authorization": f"Bearer {key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.thread_list = []

    def connect_thread(self, response_create_event, event):
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.connect(response_create_event, event))
        except:
            loop.stop()
            if len(self.thread_list) != 0:
                for t in self.thread_list:
                    del t
    
    async def connect(self, response_create_event, event):
        async with websockets.connect(REALTIME_URL, extra_headers=self.headers) as websocket:
            print('connect')
            await websocket.send(json.dumps(event))
            await websocket.send(json.dumps(response_create_event))
            while True:
                response = json.loads(await websocket.recv())
                # print(response)
                if response['type'] == 'error':
                    print(response['error'])
                    raise SystemError
                if response['type'] == 'response.done':
                    print('response finish!')
                    await websocket.close()
                    break
                if self.mode == 'text':
                    if response['type'] == 'response.text.delta':
                        self.text_output.append(response['delta'])
                        # print(f"收到的回應: {self.text_output}")
                    elif response['type'] == 'response.content_part.done':
                        self.full_text_output = response['part']['text']
                        self.text_output = []
                        # print(f"收到的回應: {self.text_output}")
                elif self.mode == 'audio':
                    if response['type'] == 'response.audio.delta':
                        self.audio_output.append(response['delta'])
                        # print(f"收到的回應: {response}")
                    elif response['type'] == 'response.content_part.done':
                        self.full_text_output = response['part']['transcript']
                        print(f"收到的回應: {self.full_text_output}")
    
    def send_text(self, text_dict:dict):
        self.input_dict = text_dict
        if self.text_stm != None:
            chat_history = self.text_stm.get()
            text_dict['chat_history'] = chat_history
        if len(self.thread_list) != 0:
            for t in self.thread_list:
                del t
        text = self.prompt.format(text_dict)
        event = {
            "type":  'conversation.item.create',
            "item": {
                "type": 'message',
                "role": "user",
                "content": [{
                    "type": 'input_text',
                    "text": text
                }]
            }
        }
        response_create_event = {
            "type": "response.create",
            "response": {
                "modalities": ['text'],
                "instructions": "請嚴格遵守文字輸入的規定，產生適當的輸出",
                "temperature": 0.6,
                "max_output_tokens": 1024
            }
        }
        t = threading.Thread(target=self.connect_thread, args=(response_create_event, event,))
        t.start()
        self.thread_list.append(t)

    def send_audio(self, audio_bytes: bytes, text_dict=None):
        if len(self.thread_list) != 0:
            for t in self.thread_list:
                del t
        self.input_dict = text_dict
        if self.text_stm != None:
            if text_dict==None:
                text_dict = {}
            chat_history = self.text_stm.get()
            text_dict['chat_history'] = chat_history
        # audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # # Resample to 24kHz mono pcm16
        # pcm_audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data
        
        # # Encode to base64 string
        # pcm_base64 = base64.b64encode(pcm_audio).decode()
        pcm_base64 = base64.b64encode(audio_bytes).decode()
        text = self.prompt.format(text_dict)
        event = {
            "type": "conversation.item.create", 
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    # {
                    #     "type": 'input_text',
                    #     "text": text
                    # },
                    {
                        "type": "input_audio", 
                        "audio": pcm_base64
                    }
                ]
            }
        }
        response_create_event = {
            "type": "response.create",
            "response": {
                "modalities": ['text', 'audio'],
                "instructions": '如果使用者沒有說話，則無需回應',
                # "請嚴格遵守文字描述的規定，回覆使用者的語音輸入，如果使用者沒有說話，則無需回應",
                "voice": "alloy",
                "output_audio_format": "pcm16",
                "temperature": 0.8,
                "max_output_tokens": 1024
            }
        }
        t = threading.Thread(target=self.connect_thread, args=(response_create_event, event,))
        t.start()
        self.thread_list.append(t)
        
    

    def listen(self, is_save=False):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
        print("Microphone activate...")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        print("Microphone close...")
        data = b''.join(frames)
        if is_save:
            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
        return data
    
    def get_full_text_output(self) -> str:
        return self.full_text_output
    
    def get_text_output(self) -> list:
        return self.text_output
    
