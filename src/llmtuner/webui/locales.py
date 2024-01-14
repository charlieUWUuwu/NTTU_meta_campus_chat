LOCALES = {
    "lang": {
        "en": {
            "label": "Lang"
        },
        "zh": {
            "label": "語言"
        }
    },
    "model_name": {
        "en": {
            "label": "Model name"
        },
        "zh": {
            "label": "模型名稱"
        }
    },
    "model_path": {
        "en": {
            "label": "Model path",
            "info": "Path to pretrained model or model identifier from Hugging Face."
        },
        "zh": {
            "label": "模型路徑",
            "info": "本機模型的檔案路徑或 Hugging Face 的模型識別碼。"
        }
    },
    "finetuning_type": {
        "en": {
            "label": "Finetuning method"
        },
        "zh": {
            "label": "微調方法"
        }
    },
    "adapter_path": {
        "en": {
            "label": "Adapter path"
        },
        "zh": {
            "label": "適配器路徑"
        }
    },
    "refresh_btn": {
        "en": {
            "value": "Refresh adapters"
        },
        "zh": {
            "value": "刷新適配器"
        }
    },
    "advanced_tab": {
        "en": {
            "label": "Advanced configurations"
        },
        "zh": {
            "label": "進階設定"
        }
    },
    "quantization_bit": {
        "en": {
            "label": "Quantization bit",
            "info": "Enable 4/8-bit model quantization (QLoRA)."
        },
        "zh": {
            "label": "量化等級",
            "info": "啟用 4/8 位元模型量化（QLoRA）。"
        }
    },
    "template": {
        "en": {
            "label": "Prompt template",
            "info": "The template used in constructing prompts."
        },
        "zh": {
            "label": "提示模板",
            "info": "建立提示詞時使用的模板"
        }
    },
    "rope_scaling": {
        "en": {
            "label": "RoPE scaling"
        },
        "zh": {
            "label": "RoPE 插值方法"
        }
    },
    "flash_attn": {
        "en": {
            "label": "Use FlashAttention-2"
        },
        "zh": {
            "label": "使用 FlashAttention-2"
        }
    },
    "shift_attn": {
        "en": {
            "label": "Use shift short attention (S^2-Attn)"
        },
        "zh": {
            "label": "使用 shift short attention (S^2-Attn)"
        }
    },
    "training_stage": {
        "en": {
            "label": "Stage",
            "info": "The stage to perform in training."
        },
        "zh": {
            "label": "訓練階段",
            "info": "目前採用的訓練方式。"
        }
    },
    "dataset_dir": {
        "en": {
            "label": "Data dir",
            "info": "Path to the data directory."
        },
        "zh": {
            "label": "數據路徑",
            "info": "資料資料夾的路徑。"
        }
    },
    "dataset": {
        "en": {
            "label": "Dataset"
        },
        "zh": {
            "label": "數據集"
        }
    },
    "data_preview_btn": {
        "en": {
            "value": "Preview dataset"
        },
        "zh": {
            "value": "預覽資料集"
        }
    },
    "preview_count": {
        "en": {
            "label": "Count"
        },
        "zh": {
            "label": "數量"
        }
    },
    "page_index": {
        "en": {
            "label": "Page"
        },
        "zh": {
            "label": "頁數"
        }
    },
    "prev_btn": {
        "en": {
            "value": "Prev"
        },
        "zh": {
            "value": "上一頁"
        }
    },
    "next_btn": {
        "en": {
            "value": "Next"
        },
        "zh": {
            "value": "下一頁"
        }
    },
    "close_btn": {
        "en": {
            "value": "Close"
        },
        "zh": {
            "value": "關閉"
        }
    },
    "preview_samples": {
        "en": {
            "label": "Samples"
        },
        "zh": {
            "label": "範例"
        }
    },
    "cutoff_len": {
        "en": {
            "label": "Cutoff length",
            "info": "Max tokens in input sequence."
        },
        "zh": {
            "label": "截斷長度",
            "info": "輸入序列分詞後的最大長度。"
        }
    },
    "learning_rate": {
        "en": {
            "label": "Learning rate",
            "info": "Initial learning rate for AdamW."
        },
        "zh": {
            "label": "學習率",
            "info": "AdamW 優化器的初始學習率。"
        }
    },
    "num_train_epochs": {
        "en": {
            "label": "Epochs",
            "info": "Total number of training epochs to perform."
        },
        "zh": {
            "label": "訓練輪數",
            "info": "需要執行的訓練總輪數。"
        }
    },
    "max_samples": {
        "en": {
            "label": "Max samples",
            "info": "Maximum samples per dataset."
        },
        "zh": {
            "label": "最大樣本數",
            "info": "每個資料集最多使用的樣本數。"
        }
    },
    "compute_type": {
        "en": {
            "label": "Compute type",
            "info": "Whether to use fp16 or bf16 mixed precision training."
        },
        "zh": {
            "label": "計算類型",
            "info": "是否啟用 FP16 或 BF16 混合精度訓練。"
        }
    },
    "batch_size": {
        "en": {
            "label": "Batch size",
            "info": "Number of samples to process per GPU."
        },
        "zh":{
            "label": "批次大小",
            "info": "每塊 GPU 上處理的樣本數。"
        }
    },
    "gradient_accumulation_steps": {
        "en": {
            "label": "Gradient accumulation",
            "info": "Number of gradient accumulation steps."
        },
        "zh": {
            "label": "梯度累積",
            "info": "梯度累積的步數。"
        }
    },
    "lr_scheduler_type": {
        "en": {
            "label": "LR Scheduler",
            "info": "Name of learning rate scheduler.",
        },
        "zh": {
            "label": "學習率調節器",
            "info": "採用的學習率調節器名稱。"
        }
    },
    "max_grad_norm": {
        "en": {
            "label": "Maximum gradient norm",
            "info": "Norm for gradient clipping.."
        },
        "zh": {
            "label": "最大梯度範數",
            "info": "用於梯度裁剪的範數。"
        }
    },
    "val_size": {
        "en": {
            "label": "Val size",
            "info": "Proportion of data in the dev set."
        },
        "zh": {
            "label": "驗證集比例",
            "info": "驗證集佔全部樣本的百分比。"
        }
    },
    "extra_tab": {
        "en": {
            "label": "Extra configurations"
        },
        "zh": {
            "label": "其它參數設定"
        }
    },
    "logging_steps": {
        "en": {
            "label": "Logging steps",
            "info": "Number of steps between two logs."
        },
        "zh": {
            "label": "日誌間隔",
            "info": "每兩次日誌輸出間的更新步數。"
        }
    },
    "save_steps": {
        "en": {
            "label": "Save steps",
            "info": "Number of steps between two checkpoints."
        },
        "zh": {
            "label": "保存間隔",
            "info": "每兩次斷點保存間的更新步數。"
        }
    },
    "warmup_steps": {
        "en": {
            "label": "Warmup steps",
            "info": "Number of steps used for warmup."
        },
        "zh": {
            "label": "預熱步數",
            "info": "學習率預熱所採用的步數。"
        }
    },
    "neftune_alpha": {
        "en": {
            "label": "NEFTune Alpha",
            "info": "Magnitude of noise adding to embedding vectors."
        },
        "zh": {
            "label": "NEFTune 噪音參數",
            "info": "嵌入向量所添加的雜訊大小。"
        }
    },
    "train_on_prompt": {
        "en": {
            "label": "Train on prompt",
            "info": "Compute loss on the prompt tokens in supervised fine-tuning."
        },
        "zh": {
            "label": "計算輸入損失",
            "info": "在監督微調時候計算輸入序列的損失。"
        }
    },
    "upcast_layernorm": {
        "en": {
            "label": "Upcast LayerNorm",
            "info": "Upcast weights of layernorm in float32."
        },
        "zh": {
            "label": "縮放歸一化層",
            "info": "將歸一化層權重縮放至 32 位元浮點數。"
        }
    },
    "lora_tab": {
        "en": {
            "label": "LoRA configurations"
        },
        "zh": {
            "label": "LoRA 參數設定"
        }
    },
    "lora_rank": {
        "en": {
            "label": "LoRA rank",
            "info": "The rank of LoRA matrices."
        },
        "zh": {
            "label": "LoRA 秩",
            "info": "LoRA 矩陣的秩。"
        }
    },
    "lora_dropout": {
        "en": {
            "label": "LoRA Dropout",
            "info": "Dropout ratio of LoRA weights."
        },
        "zh": {
            "label": "LoRA 隨機丟棄",
            "info": "LoRA 權重隨機丟棄的機率。"
        }
    },
    "lora_target": {
        "en": {
            "label": "LoRA modules (optional)",
            "info": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules."
        },
        "zh": {
            "label": "LoRA 作用模組（非必填）",
            "info": "應用 LoRA 的目標模組名稱。使用英文逗號分隔多個名稱。"
        }
    },
    "additional_target": {
        "en": {
            "label": "Additional modules (optional)",
            "info": "Name(s) of modules apart from LoRA layers to be set as trainable. Use commas to separate multiple modules."
        },
        "zh": {
            "label": "附加模組（非必填）",
            "info": "除 LoRA 層以外的可訓練模組名稱。使用英文逗號分隔多個名稱。"
        }
    },
    "create_new_adapter": {
        "en": {
            "label": "Create new adapter",
            "info": "Whether to create a new adapter with randomly initialized weight or not."
        },
        "zh": {
            "label": "新建適配器",
            "info": "是否建立一個經過隨機初始化的新適配器。"
        }
    },
    "rlhf_tab": {
        "en": {
            "label": "RLHF configurations"
        },
        "zh": {
            "label": "RLHF 參數設定"
        }
    },
    "dpo_beta": {
        "en": {
            "label": "DPO beta",
            "info": "Value of the beta parameter in the DPO loss."
        },
        "zh": {
            "label": "DPO beta 參數",
            "info": "DPO 損失函數中 beta 超參數大小。"
        }
    },
    "reward_model": {
        "en": {
            "label": "Reward model",
            "info": "Adapter of the reward model for PPO training. (Needs to refresh adapters)"
        },
        "zh": {
            "label": "獎勵模型",
            "info": "PPO 訓練中獎勵模型的轉接器路徑。 （需要刷新適配器）"
        }
    },
    "cmd_preview_btn": {
        "en": {
            "value": "Preview command"
        },
        "zh": {
            "value": "預覽命令"
        }
    },
    "start_btn": {
        "en": {
            "value": "Start"
        },
        "zh": {
            "value": "開始"
        }
    },
    "stop_btn": {
        "en": {
            "value": "Abort"
        },
        "zh": {
            "value": "中斷"
        }
    },
    "output_dir": {
        "en": {
            "label": "Output dir",
            "info": "Directory for saving results."
        },
        "zh": {
            "label": "輸出目錄",
            "info": "儲存結果的路徑。"
        }
    },
    "output_box": {
        "en": {
            "value": "Ready."
        },
        "zh": {
            "value": "準備就緒。"
        }
    },
    "loss_viewer": {
        "en": {
            "label": "Loss"
        },
        "zh": {
            "label": "損失"
        }
    },
    "predict": {
        "en": {
            "label": "Save predictions"
        },
        "zh": {
            "label": "保存預測結果"
        }
    },
    "load_btn": {
        "en": {
            "value": "Load model"
        },
        "zh": {
            "value": "載入模型"
        }
    },
    "unload_btn": {
        "en": {
            "value": "Unload model"
        },
        "zh": {
            "value": "解除安裝模型"
        }
    },
    "info_box": {
        "en": {
            "value": "Model unloaded, please load a model first."
        },
        "zh": {
            "value": "模型未載入，請先載入模型。"
        }
    },
    "system": {
        "en": {
            "placeholder": "System prompt (optional)"
        },
        "zh": {
            "placeholder": "系統提示詞（非必填）"
        }
    },
    "query": {
        "en": {
            "placeholder": "Input..."
        },
        "zh": {
            "placeholder": "輸入..."
        }
    },
    "submit_btn": {
        "en": {
            "value": "Submit"
        },
        "zh": {
            "value": "提交"
        }
    },
    "clear_btn": {
        "en": {
            "value": "Clear history"
        },
        "zh": {
            "value": "清空歷史"
        }
    },
    "max_length": {
        "en": {
            "label": "Maximum length"
        },
        "zh": {
            "label": "最大長度"
        }
    },
    "max_new_tokens": {
        "en": {
            "label": "Maximum new tokens"
        },
        "zh": {
            "label": "最大生成長度"
        }
    },
    "top_p": {
        "en": {
            "label": "Top-p"
        },
        "zh": {
            "label": "Top-p 取樣值"
        }
    },
    "temperature": {
        "en": {
            "label": "Temperature"
        },
        "zh": {
            "label": "溫度係數"
        }
    },
    "max_shard_size": {
        "en": {
            "label": "Max shard size (GB)",
            "info": "The maximum size for a model file."
        },
        "zh": {
            "label": "最大分塊大小（GB）",
            "info": "單一模型檔案的最大大小。"
        }
    },
    "export_quantization_bit": {
        "en": {
            "label": "Export quantization bit.",
            "info": "Quantizing the exported model."
        },
        "zh": {
            "label": "導出量化等級",
            "info": "量化導出模型。"
        }
    },
    "export_quantization_dataset": {
        "en": {
            "label": "Export quantization dataset.",
            "info": "The calibration dataset used for quantization."
        },
        "zh": {
            "label": "導出量化資料集",
            "info": "量化過程中使用的校準資料集。"
        }
    },
    "export_dir": {
        "en": {
            "label": "Export dir",
            "info": "Directory to save exported model."
        },
        "zh": {
            "label": "匯出目錄",
            "info": "儲存匯出模型的資料夾路徑。"
        }
    },
    "export_btn": {
        "en": {
            "value": "Export"
        },
        "zh": {
            "value": "開始匯出"
        }
    }
}


ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it firstly.",
        "zh": "任務已存在，請先中斷訓練。"
    },
    "err_exists": {
        "en": "You have loaded a model, please unload it first.",
        "zh": "模型已存在，請先卸載模型。"
    },
    "err_no_model": {
        "en": "Please select a model.",
        "zh": "請選擇模型。"
    },
    "err_no_path": {
        "en": "Model not found.",
        "zh": "模型未找到。"
    },
    "err_no_dataset": {
        "en": "Please choose a dataset.",
        "zh": "請選擇資料集。"
    },
    "err_no_adapter": {
        "en": "Please select an adapter.",
        "zh": "請選擇一個適配器。"
    },
    "err_no_export_dir": {
        "en": "Please provide export dir.",
        "zh": "請填寫匯出目錄"
    },
    "err_failed": {
        "en": "Failed.",
        "zh": "訓練出錯。"
    },
    "err_demo": {
        "en": "Training is unavailable in demo mode, duplicate the space to a private one first.",
        "zh": "展示模式不支援訓練，請先複製到私人空間。"
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "zh": "訓練中斷，正在等待線程結束…"
    },
    "info_aborted": {
        "en": "Ready.",
        "zh": "準備就緒。"
    },
    "info_finished": {
        "en": "Finished.",
        "zh": "訓練完畢。"
    },
    "info_loading": {
        "en": "Loading model...",
        "zh": "載入中……"
    },
    "info_unloading": {
        "en": "Unloading model...",
        "zh": "卸載中…"
    },
    "info_loaded": {
        "en": "Model loaded, now you can chat with your model!",
        "zh": "模型已加載，可以開始聊天了！"
    },
    "info_unloaded": {
        "en": "Model unloaded.",
        "zh": "模型已卸載。"
    },
    "info_exporting": {
        "en": "Exporting model...",
        "zh": "正在匯出模型…"
    },
    "info_exported": {
        "en": "Model exported.",
        "zh": "模型匯出完成。"
    }
}
