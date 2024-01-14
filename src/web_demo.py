from llmtuner import create_web_demo
from fastapi import FastAPI
import uvicorn
import gradio as gr

""" ori """
# def main():
#     demo = create_web_demo()
#     demo.queue()
#     demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)

# if __name__ == "__main__":
#     main()

""" fastapi """ 
# 之後改成公共的
app = FastAPI()

def create_app():
    demo = create_web_demo()
    demo.queue()
    return demo

if __name__ == "__main__":
    demo = create_app()
    app = gr.mount_gradio_app(app, demo, path="/") # 記得設定path~ 
    # uvicorn.run(app, host="localhost", port=8000)
    uvicorn.run(app)
    