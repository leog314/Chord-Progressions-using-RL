import matplotlib.pyplot as plt
from general_agent import *
from env_build import *
import time as t
from logging_init import *

# copied from ./unia

action_space = 4095 # each chord progression (2^12-1)
state_shape = 12*WINDOW_LENGTH
max_time_steps = 100 # if unnecessary, set it to a high value -> # of chord generated in one epoch
update_freq = 20 # define target-network update frequency

logger.info(f"Costum build: Chord-progressions")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Detected device: {device}")
t.sleep(1)

if device == "cpu":
    threads_num = int(input(f"Enter number of threads (available: {torch.get_num_threads()}): "))
    torch.set_num_threads(threads_num)
    logger.info(f"...Using {threads_num} cpu threads")

logger.info(f"Beginning the training process...")

epochs = 100000 # use as many epochs as you want
env = Env(reward_net, WINDOW_LENGTH)
agent = Agent(state_shape, action_space, device=device)
logger.info(f"Architecture: {agent.main_network}")

# agent.main_network = torch.load("model.pt") # optionally: load existing models
# agent.target_network = torch.load("model.pt")

for epoch in range(epochs):
    agent.main_network.train()
    agent.target_network.train()

    average_loss = 0
    average_rew = 0
    done = False

    if epoch % update_freq == 0 and epoch != 0: agent.target_network.load_state_dict(agent.main_network.state_dict())
    state = env.start_mdp()

    for step in range(max_time_steps):
        if done:
            break
        action = agent.select_action(state)
        nstate, rew = env.step(action)

        agent.replay_buffer.push(state, torch.tensor([action]), rew, nstate)

        state = nstate.clone()

        loss = agent.training_main()

        average_loss += loss
        average_rew += float(rew.clone())

    print(f"Average loss in epoch {epoch}: {average_loss/step}... and average reward in this epoch: {average_rew/step}, {agent.decay}... Got it for {step} steps")
    torch.save(agent.main_network, "generatorNet.pt")

    if epoch % 20 == 0: # some evaluation (note that due to epsilion decay, there will always be some randomness)
        generated_prog = "_START_ "
        state = env.start_mdp()

        agent.main_network.eval()

        for _ in range(4):
            env.step(2192) # Cmaj
            generated_prog += f"{index_to_name(2192)} "

        for step in range(max_time_steps):
            action = agent.select_action(state)
            nstate, rew = env.step(action)

            state = nstate.clone()

            generated_prog += f"{index_to_name(action)} ({float(rew)}) "

        print(generated_prog + "\n\n")