from dialog import SupervisedAgent, Dialog
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
# import gymnasium as gym
import itertools

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 100
possible_priorities = itertools.permutations(["Low", "Medium", "High"])
def get_reward(dialog, agent):
    if dialog != "Walk-Away":
        final_offers = dialog[-2:]
        agent_offer = final_offers[0] if list(final_offers[0].keys())[0] == agent.id else final_offers[1]
        other_offer = final_offers[0] if list(final_offers[0].keys())[0] != agent.id else final_offers[1]
        agent_offer = list(agent_offer.values())[0].split(" ")
        other_offer = list(other_offer.values())[0].split(" ")
        print(agent_offer)
        print(other_offer)
    else:
        print("Walk-Away")


def PPO_loop():
    # Load the agents
    reinforce_agent = SupervisedAgent("reinforce_agent", "flan_t5-small-casino/checkpoint-14120")
    partner_agent = SupervisedAgent("partner_agent", "flan_t5-small-casino/checkpoint-14120")
    agents = [reinforce_agent, partner_agent]
    optimizer = torch.optim.Adam(reinforce_agent.model.parameters(), lr=5e-5)
    # Load the dialog

    for epoch in range(num_epochs):
        for prio in possible_priorities:
            for partner_prio in possible_priorities:

                reinforce_agent.setPriorities(prio)
                partner_agent.setPriorities(partner_prio)
                dialog = Dialog(agents)

                # Train the dialog
                selfplay_result = dialog.selfplay()

                # Get the reward
                reward = get_reward(selfplay_result, reinforce_agent)

                # Train the model
                # optimizer.zero_grad()
                # loss = -reward
                # loss.backward()
                # optimizer.step()
                # print(loss)



PPO_loop()