import base64

class TextBuffer():
    def __init__(self, buffer_size=1)->None:
        self.buffer_size = buffer_size
        self.buffer = []
    
    def set(self, con:list) -> None:
        self.buffer.append(con)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get(self) -> str:
        text = ''
        for b in self.buffer:
            text = text + str(b) + '\n'
        return text
    
    def refresh(self) ->None:
        self.buffer = []
    
class ImageBuffer():
    def __init__(self) -> None:
        self.buffer = []

    def refresh(self)-> None:
        self.buffer = []

    def save_img(self, img) -> None:
        self.buffer.append(img)

    def get_img(self) -> list:
        return self.buffer

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')