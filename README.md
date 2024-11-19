# Llama-and-GPT
LLM python interface for API GPT and local Llama

## Installation
Only install Python for using GPT models. Install llama_cpp_python for Llama models.
Available version of llama_cpp_python on https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/
### Environment of llama_cpp_python GPU version
1. Python 3.10
3. Nvidia RTX 4090 with CUDA version 12.2
4. Visual Studio 2022

'''
set CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64"
'''
'''
python -m pip install llama-cpp-python==0.2.90 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --upgrade --force-reinstall --no-cache-dir --verbose
'''

## Support
### OpenAI GPT models
1. gpt-3.5-turbo
2. gpt-4
3. gpt-4-turbo
4. gpt-4o
5. gpt-4o-mini
6. o1-preview
7. o1-mini
### Llama 
All Llama models in .gguf. Please download from huggingface by yourself. 
### OpenAI Embeddings
1. text-embedding-3-large
2. text-embedding-3-small
3. text-embedding-ada-002

## Acknowledgments
This work is supported by National Taiwan University, Mechanical Engineering, Robotics Labotory. Thanks to Yu-Lin Zhao for debugging the GPU version of llama_cpp_python.
