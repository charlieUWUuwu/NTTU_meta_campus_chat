from llmtuner import create_web_demo

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from pyngrok import ngrok
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

# 中間件函數 (Internal Server Error = =)
# async def add_custom_header(request: Request, call_next):
#     response = await call_next(request)
#     response.headers['ngrok-skip-browser-warning'] = 5418  # 設置自定義頭部
#     return response

def create_app():
    demo = create_web_demo()
    demo.queue()
    return demo

def start_ngrok():
    url = ngrok.connect(8000)
    print("Ngrok Tunnel URL:", url)


if __name__ == "__main__":
    demo = create_app()
    app = gr.mount_gradio_app(app, demo, path="/") # 記得設定path~ 

    # 添加中間件到 FastAPI 應用
    # app.add_middleware(BaseHTTPMiddleware, dispatch=add_custom_header)

    """ ori """
    # uvicorn.run(app, host="localhost", port=8000)
    # uvicorn.run(app)

    """ ngrok """ # https://www.volcengine.com/theme/4085767-R-7-1
    start_ngrok()
    uvicorn.run(app, host="localhost", port=8000) # 啟動uvicorn伺服器


    