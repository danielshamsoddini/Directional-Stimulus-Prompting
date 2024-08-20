import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import itertools
from dialog import FlanAgent, Dialog

num_epochs = 10
possible_priorities = list(itertools.permutations(["Low", "Medium", "High"], 3))
# Reward function (your original code)
def get_reward(dialog, agent):
    if dialog != "Walk-Away":
        final_offers = dialog[-2:]
        agent_offer = final_offers[0] if list(final_offers[0].keys())[0] == agent.id else final_offers[1]
        other_offer = final_offers[0] if list(final_offers[0].keys())[0] != agent.id else final_offers[1]
        agent_offer = list(agent_offer.values())[0].split(" ")
        other_offer = list(other_offer.values())[0].split(" ")

        agent_offer = [agent_offer[1], agent_offer[3], agent_offer[5]]
        other_offer = [other_offer[1], other_offer[3], other_offer[5]]
        offer_sums = [int(agent_offer[a]) + int(other_offer[a]) for a in range(len(agent_offer))]
        if offer_sums != [3, 3, 3]:
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
        # for a in range(len(agent.priorities_quant)):
        #     final_score += 2**agent.priorities_quant[a] * 3
        return 6

# reinforce Loop (fixed)
def reinforce_loop():
    # Load the agents
    reinforce_agent = FlanAgent("reinforce_agent", "flan_t5-small-casino/checkpoint-14120")
    partner_agent = FlanAgent("partner_agent", "flan_t5-small-casino/checkpoint-14120")
    agents = [reinforce_agent, partner_agent]
    optimizer = torch.optim.Adam(reinforce_agent.model.parameters(), lr=5e-5)

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
                    if reward is None:
                        continue

                    epoch_reward += reward

                    # Train the model
                    log_probs = torch.tensor(reinforce_agent.log_probs, requires_grad=True)
                    # print(log_probs)
                    loss = -torch.sum(log_probs) * reward  # Policy Gradient loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(reinforce_agent.model.parameters(), 1.0)
                    optimizer.step()

                    print(f"Loss: {loss.item()}, Prio: {prio}, Partner Prio: {partner_prio}")

        print(f"Epoch: {epoch}, Average Reward: {epoch_reward / float(len(possible_priorities) ** 2)}")

reinforce_loop()
