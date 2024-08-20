import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import itertools
from dialog import FlanAgent, Dialog
import random

num_epochs = 50
batch_size = 6
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
    optimizer = torch.optim.SGD(reinforce_agent.model.parameters(), lr=1e-4, momentum=0.9)

    abc = list(itertools.product(possible_priorities, repeat=2))
    # random.shuffle(abc)
    base_reward = 0
    for (prio, partner_prio) in abc:

        reinforce_agent.setPriorities(prio)
        partner_agent.setPriorities(partner_prio)
        dialog = Dialog(agents)
        dialog.selfplay()
        base_reward += get_reward(dialog.dialog_history, reinforce_agent)

    print(f"Base Reward: {base_reward / float(len(possible_priorities) ** 2)}")
    with open("progress.txt", "w") as f:
        f.write(f"Base Reward: {base_reward / float(len(possible_priorities) ** 2)}\n")
    for epoch in range(num_epochs):
        epoch_reward = 0
        avg_loss = 0
        
        batch_log_probs = []
        batch_rewards = []

        # Iterate over possible priority combinations
        for i, (prio, partner_prio) in enumerate(abc):
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

                # Accumulate log probabilities and rewards
                batch_log_probs.extend(reinforce_agent.log_probs)
                batch_rewards.extend([reward] * len(reinforce_agent.log_probs))

            # When batch is full, perform optimization
            if (i + 1) % batch_size == 0:
                # Calculate batch loss
                log_probs = torch.tensor(batch_log_probs, requires_grad=True)
                rewards = torch.tensor(batch_rewards, dtype=torch.float32, requires_grad=False)
                loss = -torch.sum(log_probs * rewards)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reinforce_agent.model.parameters(), 1.0)
                optimizer.step()

                print(f"Batch Loss: {loss.item()}")
                avg_loss += loss.item()

                # Clear batch data
                batch_log_probs.clear()
                batch_rewards.clear()

        # Log epoch statistics
        epoch_str = f"Epoch: {epoch+1}, Average Loss: {avg_loss/float(len(possible_priorities) ** 2)}, Average Reward: {epoch_reward / float(len(possible_priorities) ** 2)}"
        print(epoch_str)
        with open("progress.txt", "a") as f:
            f.write(epoch_str + "\n")

    # Run a final selfplay to see the behavior after training
    abc = Dialog(agents)
    abc.selfplay()
    abc.print_dialog()
    reinforce_agent.model.save_pretrained("rl_trained", from_pt=True) 

reinforce_loop()
