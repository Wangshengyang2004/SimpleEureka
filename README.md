# Simple Eureka: Running LLM-RL generation pipeline on Omniverse Isaac Sim

## Note:
- This repo is under heavy development right now, versions maybe updated weekly, and API may change drastically
- Currently, code is heavily borrowed from NVIDIA Eureka, as we haven't made a full adaptation yet. Later, code will be more neat and tidy

## Features:
- Maintain the same logic as NVIDIA Eureka
- Support Linux and Windows thanks to Nvidia Omniverse
- Improvements to Eureka/DR-Eureka: Agents, CoT, RL resource Manual

## How to use:
1. Ensure you have installed Omniverse and Isaac Sim on Windows/Linux, remember the Path to isaac-sim-2023.1.1, we will put it into the config.yaml
2. Clone this repo, prepare for the API keys for GPT-4/3.5/LLama/Kimi, fill in the config/api folder, replace the blank of keys in the <llm_name>.yaml
3. ```python
   cd SimpleEureka
   conda create -n simple_eureka python=3.10
   conda activate simple_eureka
   pip install -r requirements.txt
   # Use Moonshot-32k
   python main_v3.py
   # Use GPT-4o
   python main_v3.py api="gpt-4o"
   # Use GPT-4
   python main_v3.py api="gpt-4"
   ```
## License:
Will be added later
## Author:
- **Shengyang Wang**, Undergraduate Student, Class of 2026, Duke Kunshan University
- **Renzo**, Undergraduate Student, Class of 2025, Duke Kunshan University
## Instructor:
- **Bing Luo**, Assistant Professor, Duke Kunshan University
- **Wenli Xiao**, Graduate School, Carnegie Mellon University

## Citation:
- No ready yet
