# LoRADiffuser

This project implements a LoRA-based fine-tuning pipeline for Stable Diffusion models to adapt them to custom datasets. It supports training LoRA modules, injecting trained weights into a diffusion model, and generating images from textual prompts. The project also includes tools for evaluating model performance using CLIP scores and other metrics.

---

## 🔧 Requirements

- Python 3.9+
- PyTorch 2.8+ with CUDA 12.8 support
- Dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```
## 📁 Project Structure
```bash
lora_project/
│
├── main.py                  # Inference, injection, and evaluation script
├── dataset.py               # Dataset preparation script
├── run.sh                   # Training and LoRA execution script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── lora_diffusion/          # LoRA library (local)
├── data/                    # Dataset folder
├── exps/                    # Output folder (LoRA weights, ignored in git)
└── .gitignore               # Git ignore file
```
## 🏗️ Setup
1. Clone the GitHub repo:
```bash
git clone https://github.com/phamkhanhquan197/LoRADiffuser.git
```
2. Create the virtual environment:
```bash
python3 -m venv LoRADiffuser
```
3. Activate the virtual environment:
```bash
cd LoRADiffuser/
source bin/activate
```
## 🖼️ Dataset Preparation
1. Download and preprocess your dataset:
```bash
python dataset.py
```
2. Dataset will be saved in:
```bash
./data/flowers/
```
## 🧠 Training LoRA Modules
1. Configure `run.sh` with your dataset and output paths:
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/flowers"
export OUTPUT_DIR="./exps/output_dsn"
```
2. Run the training scripts:
```bash
chmod +x run.sh
./run.sh
```
3. After training, LoRA weights are saved as:
```bash
./exps/output_dsn/final_lora.safetensors
```
## 🔌 Injecting LoRA into a Stable Diffusion Model
Use `main.py` to load and inject trained LoRA weights
```bash
python3 main.py
```
## 🎨 Generating Images
1. Define the prompts for generation
```bash
prompts = [
    "A beautiful flower in a fantasy garden",
    "A futuristic city at night",
    "A serene beach during sunset",
    "A robot painting a landscape",
    "A fantasy castle floating in clouds"
]
```
2. Evaluate the model and compute the CLIP score:
```bash
python3 evaluate.py
```
3. Generate and save images:
```bash
./eval_results
```
## References

[cloneofsimo/lora GitHub Repository](https://github.com/cloneofsimo/lora)
