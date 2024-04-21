# About config file
create a config.yaml under the SimpleEureka folder
```yaml
# isaacsim path, gym environment, python path
gym:
  isaacsimpath : "/home/simonwsy/.local/share/ov/pkg/isaac_sim-2023.1.1"
  task : "Crazyflie"
  pythonpath : "~/.local/share/ov/pkg/isaac_sim-*/python.sh"
  scriptpath : "scripts/rlgames_train.py"
  headless : true
  extra : "--env-config-path ./config.yaml"
  multirun : false

# output path for simple eureka
output:
  path : "./results/{date}"
  overwrite : "/home/simonwsy/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/crazyflie.py"

generation:
  number: 2
  epochs: 5

defaults:
  - _self_
  - api: moonshot
```

```yaml
# In ./api folder
name: "moonshot"
model: "moonshot-v1-32k"
url : "https://api.moonshot.cn/v1"
key : "Your key here"
temperature : 1.0
max_tokens : 2200

```
```bash
H:.
│  .DS_Store
│  .gitignore
│  app.py
│  main.py
│  README.md
│  requirements.txt
│
├─backup
│      main.py
│
├─config
│  │  config.yaml
│  │  config_example.yaml
│  │
│  └─api
│          claude3.yaml
│          custom_llm.yaml
│          gpt-4.yaml
│          llama3.yaml
│          mixtral8x7b.yaml
│          moonshot.yaml
│
├─input
│      crazyflie.py
│      crazyflie.yaml
│      crazyflie_human_diff.py
│      task_description.txt
│      updated_crazyflie.py
│
├─outputs
│  └─2024-04-20
│      ├─00-03-55
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─00-05-31
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─00-06-30
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─00-10-53
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─00-17-07
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─00-20-07
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─00-22-23
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─00-24-46
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─10-51-42
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─10-56-16
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─10-59-48
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─13-24-29
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─13-33-32
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─14-02-11
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      ├─14-02-45
│      │  │  main.log
│      │  │
│      │  └─.hydra
│      │          config.yaml
│      │          hydra.yaml
│      │          overrides.yaml
│      │
│      └─14-10-34
│          │  main.log
│          │
│          └─.hydra
│                  config.yaml
│                  hydra.yaml
│                  overrides.yaml
│
├─prompts
│      code_feedback.txt
│      execution_error_feedback.txt
│      initial.txt
│      must_do.txt
│      paraphrase.py
│      tips.txt
│
├─results
│  │  full_prompt.txt
│  │
│  ├─2024-04-20_00-20-06
│  │      response_0.txt
│  │
│  ├─2024-04-20_00-22-23
│  │      response_0.txt
│  │      response_1.txt
│  │      updated_crazyflie_0.py
│  │      updated_crazyflie_1.py
│  │
│  ├─2024-04-20_00-24-46
│  │      response_0.txt
│  │      response_1.txt
│  │      updated_crazyflie_0.py
│  │      updated_crazyflie_1.py
│  │
│  ├─2024-04-20_10-51-42
│  │      response_0.txt
│  │      response_1.txt
│  │      updated_crazyflie_1.py
│  │
│  ├─2024-04-20_10-56-16
│  │      output.log
│  │      response_0.txt
│  │      response_1.txt
│  │
│  ├─2024-04-20_10-59-32
│  │      output.log
│  │
│  ├─2024-04-20_10-59-48
│  │      output.log
│  │      response_0.txt
│  │      response_1.txt
│  │      updated_crazyflie_0.py
│  │      updated_crazyflie_1.py
│  │
│  ├─2024-04-20_13-24-29
│  │      output.log
│  │      response_0.txt
│  │      response_1.txt
│  │      updated_crazyflie_0.py
│  │      updated_crazyflie_1.py
│  │
│  ├─2024-04-20_13-29-05
│  │      output.log
│  │
│  ├─2024-04-20_13-33-31
│  │      output.log
│  │      response_0.txt
│  │      response_1.txt
│  │      updated_crazyflie_0.py
│  │      updated_crazyflie_1.py
│  │
│  ├─2024-04-20_14-02-10
│  │      output.log
│  │
│  ├─2024-04-20_14-02-45
│  │      output.log
│  │      response_0.txt
│  │      response_1.txt
│  │      updated_crazyflie_0.py
│  │      updated_crazyflie_1.py
│  │
│  └─2024-04-20_14-10-34
│          output.log
│          response_0.txt
│          response_1.txt
│          updated_crazyflie_0.py
│          updated_crazyflie_1.py
│
└─utils
    │  create_task.py
    │  extract_task_code.py
    │  file_utils.py
    │  llm_endpoint.py
    │  misc.py
    │  prune_env.py
    │  prune_env_dexterity.py
    │  prune_env_isaac.py
    │  simple_eureka.py
    │
    └─__pycache__
            extract_task_code.cpython-310.pyc
            file_utils.cpython-310.pyc
            simple_eureka.cpython-310.pyc```