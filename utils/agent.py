import hydra
from utils.misc import file_to_string
import openai
import os
from loguru import logger

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.EUREKA_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.actor_dir = f"{self.EUREKA_ROOT_DIR}/prompts/actor"
        self.critic_dir = f"{self.EUREKA_ROOT_DIR}/prompts/critic"
    
    def critic_agent(self):
            """
            Set up the critic agent for the Eureka system
            This agent will explain the task description in detail and
            guide the reward function engineer on performance,
            creativity, and robustness of the reward function
            """
            cfg = self.cfg

            name = cfg.critic_agent.name
            fellow_name = cfg.actor_agent.name
            # Load prompt for critic agent
            critic_prompt = file_to_string(f"{self.critic_dir}/user.txt")
            task_description: str = file_to_string(f"{self.EUREKA_ROOT_DIR}/input/task_description.txt")
            system_instruction = file_to_string(f"{self.critic_dir}/system.txt")
            system_instruction = system_instruction.format(
                name=name,
                task_description=task_description, 
                fellow_name=fellow_name
            )
            messages = [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": critic_prompt
                }
            ]
            logger.info(f"Critic Agent: {system_instruction} \n {critic_prompt}")
            client = openai.OpenAI(api_key=cfg.api.key, base_url=cfg.api.url)
            response_cur = client.chat.completions.create(
                                model=cfg.api.model,
                                messages=messages,
                                temperature=cfg.api.temperature,
                                max_tokens=1000,
                                n=1
                            )
            self.detailed_task_description: str = response_cur.choices[0].message.content
            logger.info(f"Critic Agent: {self.detailed_task_description}")
            logger.info(f"\n Prompt Token Cost: {response_cur.usage.prompt_tokens} \n Response Token Cost: {response_cur.usage.completion_tokens}")

    def actor_agent(self):
        """    
        # Load text from the prompt file, Assemble prompt for actor agent
        """
        initial_system = file_to_string(f"{self.actor_dir}/initial_system.txt")
        api_doc = file_to_string(f"{self.actor_dir}/api_doc.txt")
        external_source = file_to_string(f"{self.actor_dir}/external_source.txt")
        code_output_tip = file_to_string(f"{self.actor_dir}/code_output_tip.txt")
        initial_user = file_to_string(f"{self.actor_dir}/initial_user.txt")
        human_code_diff = file_to_string(f"{self.EUREKA_ROOT_DIR}/input/crazyflie_human_diff.py")
        task_obs_code_string = file_to_string(f"{self.EUREKA_ROOT_DIR}/input/crazyflie.py")
        env_config = file_to_string(f"{self.EUREKA_ROOT_DIR}/input/crazyflie.yaml")
        must_do = file_to_string(f"{self.actor_dir}/must_do.txt")
        final = "Add a sign for end of your code, when you finish the is_done part: #END\n"

        # Assemble the full prompt
        initial_user = initial_user.format(
            task_obs_code_string=task_obs_code_string,
            optimized_task_description=self.detailed_task_description,
            code_output_tip=code_output_tip,
            human_code_diff=human_code_diff,
            env_config=env_config,
            final=final,
            must_do=must_do,
            api_doc=api_doc,
            external_source=external_source
        )

        self.messages = [
            {
                "role": "system",
                "content": initial_system
            },
            {
                "role": "user",
                "content": initial_user
            }
        ]

    def message(self):
         """
         Assemble the full prompt
         """
         self.critic_agent()
         self.actor_agent()
         return self.messages

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    agent = Agent(cfg=cfg)
    agent.critic_agent()
    agent.actor_agent()
    # print(agent.message())
    print(agent.detailed_task_description)
if __name__ == "__main__":
    main()