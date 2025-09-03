# 🔮 Fine-Tuning LLM with LoRA (Qwen 0.5B-Instruct)

This project demonstrates **fine-tuning a Large Language Model (LLM)** using **LoRA (Low-Rank Adaptation)** for parameter-efficient training.  
The model is based on **[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen)** and is fine-tuned on a **custom JSON dataset** created by me.

---

## ✨ Project Overview
- Implemented **parameter-efficient fine-tuning (PEFT)** using **LoRA**.  
- Used **custom JSON dataset** (`prabhat_dat.json`) containing **prompt-completion pairs** for training.  
- Applied **tokenization, truncation, and padding** to preprocess the data.  
- Trained using **Hugging Face `Trainer` API** with **PyTorch** backend.  
- Deployed the fine-tuned model for **inference** with Hugging Face `pipeline`.  

---

## 📂 Repository Structure
```
llm_fine_tuning.ipynb      # Main notebook with training & fine-tuning code
prabhat_dat.json           # Custom dataset in JSON format (prompt + completion pairs)
trainer_output/            # Training logs and outputs
prabhat_llma/              # Saved fine-tuned model + tokenizer
```

---

## 🛠️ Tech Stack
- **Python**  
- **Hugging Face Transformers**  
- **PEFT (LoRA)**  
- **PyTorch**  
- **Datasets (Hugging Face)**  

---

## ⚙️ How It Works

### 1. **Dataset Preparation**
- Created a custom dataset `prabhat_dat.json` in the following format:
```json
{
  "prompt": "Who is Prabhat Kumar?",
  "completion": "Prabhat Kumar is a Data Scientist with expertise in AI/ML."
}
```
- Loaded dataset with Hugging Face `datasets.load_dataset("json")`.  

---

### 2. **Preprocessing**
- Tokenized data using **Qwen tokenizer**.  
- Applied:
  - **Truncation** to max length = 128  
  - **Padding** to uniform size  
  - Labels created as a copy of input IDs  

---

### 3. **LoRA Fine-Tuning**
- Loaded **Qwen2.5-0.5B-Instruct** with `AutoModelForCausalLM`.  
- Applied **LoRA adapters** on attention projection layers (`q_proj`, `k_proj`, `v_proj`).  
- Used **fp16 (half precision)** training for speed on GPU.  

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"]
)
```

---

### 4. **Training**
- Configured training with Hugging Face `Trainer`:  
  - Epochs: **12**  
  - Learning rate: **0.001**  
  - Mixed precision: **fp16**  
  - Logging steps: **25**  

- Saved fine-tuned model and tokenizer to `./prabhat_llma`.  

---

### 5. **Inference**
- Loaded fine-tuned model with Hugging Face `pipeline`:  
```python
ask_llm = pipeline(
    model="./prabhat_llma",
    tokenizer="./prabhat_llma",
    device="cuda",
    torch_dtype=torch.float16
)
ask_llm("Who is Prabhat Kumar?")[0]['generated_text']
```

- Generates **custom responses** based on fine-tuned dataset.  

---

## 🚀 Key Learnings
- **LoRA** makes fine-tuning efficient by only updating a fraction of parameters.  
- **JSON dataset format** is flexible and works well for instruction-based fine-tuning.  
- **Trainer API** simplifies the training loop with logging, checkpointing, and fp16 support.  
- **Qwen 0.5B** is lightweight and runs well on limited hardware (single GPU).  

---

## 📦 Installation
Install dependencies:
```bash
pip install --upgrade transformers datasets accelerate torch torchvision peft pillow
```

---

## 🤝 Contributing
Contributions, suggestions, and improvements are welcome! 🚀  

---

## 📄 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  
