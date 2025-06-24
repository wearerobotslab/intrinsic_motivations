import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an Intrisically motivated agent")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="hk", help="Agent [hk, emp, tipi].")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedEnvCfg
from homeokinesis import Homeokinesis
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg, load_cfg_from_registry

def main():
    """Play with an intrinsically motivated agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.episode_length_s = 100
    agent_cfg = load_cfg_from_registry(args_cli.task, "im_control_hk_cfg_entry_point")

    # create isaac environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    agent_cfg['params']['num_inputs'] = env.observation_manager.compute_group('proprioception').shape[1]
    agent_cfg['params']['num_outputs'] = env.action_manager.action.shape[1]
    agent_cfg['params']['num_envs'] = env.num_envs
    im_agent = Homeokinesis(agent_cfg['params'])

    # reset environment
    x = env.observation_manager.compute_group('proprioception')
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            x = env.observation_manager.compute_group('proprioception')
            actions = im_agent.step(x)
            # env stepping
            obs, _, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
