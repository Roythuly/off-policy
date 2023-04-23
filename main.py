import os
import torch
import gym
import numpy as np
from utilis.config import ARGConfig
from utilis.default_config import default_config
from model.algo import SAC, TD3
from utilis.Replaybuffer import ReplayMemory
import datetime
import itertools
from copy import copy
from torch.utils.tensorboard import SummaryWriter
import shutil


def train_loop(config, msg = "default"):
    # set seed
    env = gym.make(config.env_name)
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Agent
    if config.algo == "SAC":
        agent = SAC(env.observation_space.shape[0], env.action_space, config)
    elif config.algo == "TD3":
        agent = TD3(env.observation_space.shape[0], env.action_space, config)

    result_path = './results/{}/{}/{}/{}_{}_{}_{}'.format(config.env_name, config.algo, msg, 
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                      config.policy, config.seed, 
                                                      "autotune" if config.automatic_entropy_tuning else "")

    checkpoint_path = result_path + '/' + 'checkpoint'
    
    # training logs
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open(os.path.join(result_path, "config.log"), 'w') as f:
        f.write(str(config))

    writer = SummaryWriter(result_path)
    shutil.copytree('.', result_path + '/code', ignore=shutil.ignore_patterns('results'))

    # memory
    memory = ReplayMemory(config.replay_size, config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_reward = -1e6
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if config.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > config.batch_size:
                # Number of updates per step in environment
                if config.algo == "SAC":
                    for i in range(config.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, config.batch_size, updates)

                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        updates += 1
                elif config.algo == "TD3":
                    for i in range(config.updates_per_step):
                        # Update parameters of all the networks
                        critic_loss, policy_loss = agent.update_parameters(memory, config.batch_size, updates)

                        writer.add_scalar('loss/critic', critic_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > config.num_steps:
            break

        writer.add_scalar('train/reward', episode_reward, total_numsteps)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        # test agent
        if i_episode % config.eval_episodes == 0 and config.eval is True:
            avg_reward = 0.
            # avg_success = 0.
            for _  in range(config.eval_episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
                # avg_success += float(info['is_success'])
            avg_reward /= config.eval_episodes
            # avg_success /= config.eval_episodes
            if avg_reward >= best_reward and config.save is True:
                best_reward = avg_reward
                agent.save_checkpoint(checkpoint_path, 'best')

            writer.add_scalar('test/avg_reward', avg_reward, total_numsteps)
            # writer.add_scalar('test/avg_success', avg_success, total_numsteps)

            print("----------------------------------------")
            print("Env: {}, Test Episodes: {}, Avg. Reward: {}".format(config.env_name, config.eval_episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close() 


if __name__ == "__main__":
    arg = ARGConfig()
    arg.add_arg("env_name", "HalfCheetah-v2", "Environment name")
    arg.add_arg("device", 0, "Computing device")
    arg.add_arg("algo", "SAC", "choose SAC or TD3")
    arg.add_arg("policy", "Gaussian", "Policy Type: Gaussian | Deterministic (default: Gaussian)")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("start_steps", 10000, "Number of start steps")
    arg.add_arg("automatic_entropy_tuning", False, "Automaically adjust α (default: False)")
    arg.add_arg("seed", 123456, "experiment seed")
    arg.parser()

    config = default_config
    config.update(arg)

    print(f">>>> Training {config.algo} on {config.env_name} environment, on {config.device}")
    train_loop(config, msg=config.tag)