# About config file
create a config.yaml under the SimpleEureka folder
```yaml
# isaacsim path, gym environment, python path
gym:
  isaacsimpath : "/home/isaac/isaac-sim-2021.1.0.0"
  gymenv : "Crazyflie"
  pythonpath : "/home/isaac/isaac-sim-2021.1.0.0/python_samples"

# output path for simple eureka
output:
  path : "/home/isaac/isaac-sim-2021.1.0.0/python_samples/gym/output"

api:
  name: "moonshot"
  model: "moonshot-v1-32k"
  url : "https://api.moonshot.cn/v1"
  key : "Your key here"
```