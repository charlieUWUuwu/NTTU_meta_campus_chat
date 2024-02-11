# 將 xlsx 文件製作成 Document 樣子
from langchain.docstore.document import Document
import pandas as pd
import os

"""
    將 xlsx, csv 文件製作成 Document 樣子

    Args:
        file_paths: list of file path
    Returns:
        texts: list of Document
"""
def process(file_paths):
    data = pd.DataFrame()
    for path in file_paths:
        _, ext = os.path.splitext(path)
        if ext == ".xlsx":
            temp = pd.read_excel(path)
        elif ext == ".csv":
            temp = pd.read_csv(path, encoding='Big5')
        temp = temp.dropna(subset=[col for col in temp.columns if col != 'link']) # 除了link外，包含 NaN 的行都刪掉
        data = pd.concat([data, temp], ignore_index=True)

    """ 製作成 Document """
    lists_of_columns = [data[col].tolist() for col in data.columns]
    titles, contents, links = lists_of_columns
    texts = []
    for i in range(len(data)):
        # page_content 為 [標題 + 段落內容]
        d = Document(page_content=titles[i]+"-"+contents[i], metadata={'source': titles[i], 'link': links[i]})
        texts.append(d)

    return texts