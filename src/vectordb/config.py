# """ 由於執行腳本不同，故使用絕對位置 """
# # 取得目前腳本檔案的絕對路徑
# current_script_path = os.path.abspath(__file__)
# print("current : ", current_script_path) # /home/p76124388/llama_factory/LLaMA-Factory/src/vectordb/config.py

# # 取得目前腳本所在的目錄
# directory_path = os.path.dirname(current_script_path)
# print("directory_path : ", directory_path) # /home/p76124388/llama_factory/LLaMA-Factory/src/vectordb

# # 建構 config.json 檔案的完整路徑
# config_file_path = os.path.join(directory_path, 'config.json')

# with open(config_file_path, "r", encoding="utf8") as f:
#     CONFIG = json.load(f)


# vector database settings
HUGGINGFACE_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "basic"

# general
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")