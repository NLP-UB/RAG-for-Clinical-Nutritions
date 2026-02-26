import requests

class Generator:
    def __init__(self, model_name='llama3.2:3b', base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, context, question):
        prompt = f"AJawab pertanyaan berdasarkan konteks berikut.\n Konteks: {context}\nPertanyaan: {question}\nJawaban:"
        payload={
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json = payload,
            timeout=600
        )
        
        response.raise_for_status()
        return response.json()["response"].strip()
