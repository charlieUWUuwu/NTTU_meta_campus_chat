![# LLaMA Factory](assets/logo.png)

> [!NOTE]
> This repo is forcked from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

Please refer to [data/README.md](data/README.md) for details.



## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- sentencepiece, protobuf and tiktoken
- jieba, rouge-chinese and nltk (used at evaluation and predict)
- gradio and matplotlib (used in web UI)
- uvicorn, fastapi and sse-starlette (used in API)

### Hardware Requirement

| Method | Bits |   7B  |  13B  |  30B  |   65B  |   8x7B |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ |
| Full   |  16  | 160GB | 320GB | 600GB | 1200GB | 1000GB |
| Freeze |  16  |  20GB |  40GB | 120GB |  240GB |  200GB |
| LoRA   |  16  |  16GB |  32GB |  80GB |  160GB |  120GB |
| QLoRA  |   8  |  10GB |  16GB |  40GB |   80GB |   80GB |
| QLoRA  |   4  |   6GB |  12GB |  24GB |   48GB |   32GB |

## Getting Started
1. install the requirements
```bash
conda create --name myenv python=3.10
pip install -r requirements.txt
```

2. (optional) login HuggingFace account
Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.
```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

3. run the web page
```bash
python src/web_demo.py \
    --model_name_or_path bigscience/bloomz-1b1 \
    --adapter_name_or_path unmerge_model/bloomz-1b1_NTTU \
    --template alpaca \
    --finetuning_type lora
```

## how to maintain this repo
每次增修內容前請依循下列流程進行：
1. Pull origin/develop 最新版本
    ```shell
    $ git pull origin develop
    ```
2. 在 local 新增 branch 並切換
    ```shell
    $ git checkout -b <NEW_BRANCH_NAME>
    ```
3. 編輯完成後進行 commit
    ```shell
    $ git add .
    $ git commit -m "COMMIT_MSG"
    ```
4. 回到 develop 再次獲取 origin/develop 的最新版本、與自己的修正合併並修正出現的 conflict
    ```shell
    $ git checkout develop
    $ git pull
    $ git checkout <NEW_BRANCH_NAME>
    $ git rebase develop
    ```
5. 將新 branch 的修正與 develop 合併並 push 到 Github
    ```shell
    $ git checkout develop
    $ git merge <NEW_BRANCH_NAME>
    $ git push
    ```