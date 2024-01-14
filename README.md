![# LLaMA Factory](assets/logo.png)

> [!NOTE]
> This repo is forcked from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

Please refer to [data/README.md](data/README.md) for details.

Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

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






## Citation

If this work is helpful, please kindly cite as:

```bibtex
@Misc{llama-factory,
  title = {LLaMA Factory},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/LLaMA-Factory}},
  year = {2023}
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.
