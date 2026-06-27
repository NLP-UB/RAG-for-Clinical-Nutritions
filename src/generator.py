import requests

class Generator:
    def __init__(self, model_name='llama3.2:3b', base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def _build_prompt_rag(self, context: str, question: str) -> str:
        return f"""Anda adalah asisten gizi klinis yang WAJIB menjawab hanya berdasarkan KONTEXT.

        Aturan ketat:
        1) Gunakan HANYA informasi pada KONTEXT.
        2) Dilarang menambah pengetahuan dari luar, asumsi, atau tebakan.
        3) Jika informasi tidak ada di KONTEXT, tulis tepat: "Tidak ditemukan pada konteks."
        4) Jangan ubah angka, satuan, atau nilai lab dari konteks.
        5) Jawaban ringkas dan langsung ke pertanyaan.

        KONTEKS:
        {context}

        PERTANYAAN:
        {question}
        
        JAWABAN:
        """

    def _build_prompt_non_rag(self, question: str) -> str:
        return f"""Anda adalah asisten gizi klinis.

        Jawab pertanyaan dengan ringkas, jelas, dan terstruktur.
        Jika ada ketidakpastian, beri catatan singkat keterbatasan jawaban.
        Fokus ke poin praktis klinis.

        PERTANYAAN:
        {question}
        
        JAWABAN:
        """

    def generate(self, context, question, use_rag=False):
        if use_rag:
            prompt = self._build_prompt_rag(context=context, question=question)
        else:
            prompt = self._build_prompt_non_rag(question=question)
            
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
