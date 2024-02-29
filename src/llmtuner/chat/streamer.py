# ref : https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L24
from queue import Queue

class ChatgptStreamer:
    def __init__(self, timeout: float=3):
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def _on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def model_generate(self, model, system, query):
        for chunk in model.stream(f"請你根據以下參考資料回答台東大學的問題，且回答時請以「元大學」來稱呼台東大學。\n相關資料:{system}\n問題:{query}"):
            self._on_finalized_text(chunk.content)
        self._on_finalized_text(self.stop_signal, True)
    
    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value