# Fine-tuning LLaMA and Implementing on NVIDIA Jetson Nano
This repository contains the code and documentation for my Final Year Project at City University of Hong Kong. The project demonstrates a complete end-to-end workflow: fine-tuning a Large Language Model (LLM) for a specific language, optimizing it for efficiency, and deploying it on resource-constrained edge hardware.

![alt text](https://img.shields.io/badge/Python-3.10+-blue.svg)

![alt text](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=flat&logo=PyTorch&logoColor=white)

![alt text](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)

## Project Overview
The primary goal of this project was to take a pre-trained Large Language Model, which is LLaMA-3.2-1B-Instruct, and specialize it for conversational tasks in Traditional Chinese. Following the fine-tuning process, the model was quantized and successfully deployed on an NVIDIA Jetson Nano, showcasing a practical application of Edge AI.

## Key Features
- **Model Fine-tuning**: Used LoRA to efficiently fine-tune the unsloth/Llama-3.2-1B-Instruct model.
- **Custom Data Pipeline**: Utilized a 100,000-sample subset of the lchakkei/OpenOrca-Traditional-Chinese dataset. A custom data formatting function was implemented to map the dataset's unique structure (system_prompt, question, response) to the required Llama-3 chat template.
- **Model Quantization for Edge**: Converted the fine-tuned model into the GGUF format (Q4_K_M and Q8_0), a critical step for reducing the model's size and enabling it to run on low-power devices.
- **Edge AI Deployment**: Successfully ran the quantized Q4_K_M.gguf and Q8_0.gguf models on an NVIDIA Jetson Nano using the llama.cpp inference engine, proving the viability of the entire pipeline from cloud training to edge execution.

## Tech Stack
- **Programming & ML**: Python, PyTorch, Jupyter Notebook
- **LLM Libraries**: Hugging Face transformers, datasets, peft, trl
- **Optimization Framework**: Unsloth
- **Deployment Hardware**: NVIDIA Jetson Nano 4GB
- **Inference Engine**: llama.cpp

## Project Workflow
1. **Environment Setup**: Configured a GPU-accelerated environment (Google Colab with A100 GPU) for model training.
2. **Data Preparation**: Loaded the OpenOrca-Traditional-Chinese dataset. Implemented the custom formatting_prompts_func to correctly structure the system, user, and assistant turns into the Llama-3 chat format, ensuring optimal training.
3. **LoRA Fine-tuning**: Configured and ran the SFTTrainer for a full epoch over 100,000 examples. Used the train_on_responses_only utility to focus the model's learning on generating high-quality assistant responses.
4. **Quantization & Export**: After training, the fine-tuned LoRA adapters were merged with the base model. The full-precision model was then quantized and saved into multiple GGUF formats (Q8_0 and Q4_K_M) for deployment.
5. **Edge Deployment & Inference**: Transferred the final, compact unsloth.Q4_K_M.gguf model file to an NVIDIA Jetson Nano. The llama.cpp library was compiled on the Jetson, and the model was successfully loaded to perform inference directly on the edge device.

## Challenges & Solutions
- **Challenge 1: Adapting a generic dataset for a specific chat format.**
  - **Solution**: The OpenOrca-Traditional-Chinese dataset has a system_prompt, question, and response structure. I wrote a Python function to iterate through these columns and correctly apply the tokenizer.apply_chat_template method.
- **Challenge 2: Running a modern LLM on the Jetson Nano's limited resources.**
  - **Solution**: This was addressed by selecting a small-but-capable 1B parameter model and, crucially, using post-training quantization. By converting the model to the Q4_K_M and Q8_0 GGUF format, the memory footprint was reduced from over 2GB (for FP16) to under 1GB, allowing it to fit within the Jetson's RAM. The highly optimized llama.cpp engine ensured that inference was possible at an interactive speed.
