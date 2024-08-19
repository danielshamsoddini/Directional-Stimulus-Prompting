from dialog import FlanAgent, Dialog
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
# import gymnasium as gym
import itertools
import logging
import copy

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 100
possible_priorities = list(itertools.permutations(["Low", "Medium", "High"]))
def get_reward(dialog, agent):
    if dialog != "Walk-Away":
        final_offers = dialog[-2:]
        agent_offer = final_offers[0] if list(final_offers[0].keys())[0] == agent.id else final_offers[1]
        other_offer = final_offers[0] if list(final_offers[0].keys())[0] != agent.id else final_offers[1]
        agent_offer = list(agent_offer.values())[0].split(" ")
        other_offer = list(other_offer.values())[0].split(" ")
        print(agent_offer)
        print(other_offer)
        #['Submit-Deal', '1', 'Firewood', '3', 'Water', '2', 'Food']
        agent_offer = [agent_offer[1], agent_offer[3], agent_offer[5]]
        other_offer = [other_offer[1], other_offer[3], other_offer[5]]
        offer_sums = [int(agent_offer[a]) + int(other_offer[a]) for a in range(len(agent_offer))]
        if offer_sums != [3,3,3]:
            print("FAILURE, OFFERS DO NOT SUM TO 3")
            return None
        else:
            print("VALID OFFER")
            final_score = 0
            for a in range(len(agent_offer)):
                final_score += 2**agent.priorities_quant[a] * int(agent_offer[a])
            return final_score
    else:
        print("Walk-Away")
        final_score = 0
        for a in range(len(agent.priorities_quant)):
            final_score += 2**agent.priorities_quant[a] * 3
        return final_score
        



def PPO_loop():
    # Load the agents
    reinforce_agent = FlanAgent("reinforce_agent", "flan_t5-small-casino/checkpoint-14120")
    partner_agent = FlanAgent("partner_agent", "flan_t5-small-casino/checkpoint-14120")
    agents = [reinforce_agent, partner_agent]
    optimizer = torch.optim.Adam(reinforce_agent.model.parameters(), lr=5e-5)
    # Load the dialog

    for epoch in range(num_epochs):
        epoch_reward = 0
        for prio in possible_priorities:
            for partner_prio in possible_priorities:

                reinforce_agent.setPriorities(prio)
                partner_agent.setPriorities(partner_prio)
                dialog = Dialog(agents)

                # Train the dialog
                selfplay_result = dialog.selfplay()

                # Get the reward
                if selfplay_result:
                    reward = get_reward(selfplay_result, reinforce_agent)
                    epoch_reward+=reward
                    # Train the model
                    optimizer.zero_grad()
                    loss = -reward
                    loss.backward()
                    optimizer.step()
                    print(loss)
                    print(prio,partner_prio)

        print(epoch)
        print(epoch_reward/float(len(possible_priorities)**2))


PPO_loop()