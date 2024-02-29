# 將 xlsx 文件製作成 Document 樣子
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, BSHTMLLoader, UnstructuredHTMLLoader
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import os

def _get_loader(file_type, file_path):
    if file_type == ".pdf":
        return PyPDFLoader(file_path)
    elif file_type == ".docx" or file_type == ".doc":
        return Docx2txtLoader(file_path)
    elif file_type == ".xlsx" or file_type == ".csv":
        return TableProcessor(file_path)
    elif file_type == ".html":
        return UnstructuredHTMLLoader(file_path)
    else:
        print("file type not supported")
        return None


def get_split_data(file_paths: List[str]) -> List[Document]:
    texts = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        _, file_extension = os.path.splitext(file_path)
        loader = _get_loader(file_extension, file_path)

        if loader is not None:
            if isinstance(loader, TableProcessor):
                temp_texts = loader.process()
                for text in temp_texts:
                    texts.append(text)
            else:
                doc = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=200,
                    length_function=len,
                )
                temp_texts = splitter.split_documents(doc)

                for i in range(len(temp_texts)):
                    # 修改 metadata 的 source 成檔名
                    temp_texts[i].metadata["source"] = file_name 
                    
                    # page_content 為 [標題 + 段落內容]
                    temp_texts[i].page_content = file_name + "-" + temp_texts[i].page_content
                    texts.append(temp_texts[i])
        else:
            raise Exception("file type not supported")
        
    return texts
    


"""
    將 xlsx, csv 文件製作成 Document 樣子
    ** 其中表格需要有 titles, contents, links 三個 columns

    Args:
        file_path: file path
    Returns:
        texts: list of Document
"""
class TableProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        data = pd.DataFrame()
        _, ext = os.path.splitext(self.file_path)
        if ext == ".xlsx":
            data = pd.read_excel(self.file_path)
        elif ext == ".csv":
            data = pd.read_csv(self.file_path, encoding='Big5')
        data = data.dropna(subset=[col for col in data.columns if col != 'link']) # 除了link外，包含 NaN 的行都刪掉

        """ 製作成 Document """
        lists_of_columns = [data[col].tolist() for col in data.columns]
        titles, contents, links = lists_of_columns
        texts = []
        for i in range(len(data)):
            # page_content 為 [標題 + 段落內容]
            d = Document(page_content=titles[i]+"-"+contents[i], metadata={'source': titles[i], 'link': links[i]})
            texts.append(d)

        return texts
    

# def table_process(file_paths):
#     data = pd.DataFrame()
#     for path in file_paths:
#         _, ext = os.path.splitext(path)
#         if ext == ".xlsx":
#             temp = pd.read_excel(path)
#         elif ext == ".csv":
#             temp = pd.read_csv(path, encoding='Big5')
#         temp = temp.dropna(subset=[col for col in temp.columns if col != 'link']) # 除了link外，包含 NaN 的行都刪掉
#         data = pd.concat([data, temp], ignore_index=True)

#     """ 製作成 Document """
#     lists_of_columns = [data[col].tolist() for col in data.columns]
#     titles, contents, links = lists_of_columns
#     texts = []
#     for i in range(len(data)):
#         # page_content 為 [標題 + 段落內容]
#         d = Document(page_content=titles[i]+"-"+contents[i], metadata={'source': titles[i], 'link': links[i]})
#         texts.append(d)

#     return texts