from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import random
import torch
from torch.nn.functional import log_softmax
import itertools
import numpy as np

starter_generation_params = {
    "max_new_tokens": 100,
    "do_sample": True, 
    "top_k": 50,
    "top_p": 0.92,
    "return_dict_in_generate": True,
    "output_logits": True
}
generation_params = {
    "max_new_tokens": 100,
    "return_dict_in_generate": True,
    "output_scores": True
}

debug = False

class FlanAgent:
    def __init__(self, id, model_dir):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.priorities = "Priorities: Low Firewood Medium Water High Food "
        self.priorities_quant = [0, 1, 2]
        self.id = id
        self.log_probs = []

    def respond(self, text, starter):
        if debug:
            print(text)
        gen_p = generation_params if self.id == "reinforce_agent" else starter_generation_params
        inputs = self.tokenizer(["Continue writing the following text.\n\n" + self.priorities + text], return_tensors="pt")
        outputs = self.model.generate(**inputs, **gen_p)

        log_probs = []
        chosen_seq = outputs['sequences'][0]
        out = outputs['scores'] if self.id == "reinforce_agent" else outputs['logits']
        for i, logits in enumerate(out):
            probs = log_softmax(logits, dim=-1)  # Get log-softmax over logits
            token_id = chosen_seq[i]  # Get token id
            log_probs.append(probs[0, token_id].item())  # Get log-prob of generated token
        self.log_probs.extend(log_probs)
        return self.tokenizer.decode(chosen_seq, skip_special_tokens=True)

    def initialize(self, priorities):
        low_to_high = {"Low": 0, "Medium": 1, "High": 2}
        self.priorities = f"Priorities: {priorities[0]} Firewood {priorities[1]} Water {priorities[2]} Food "
        self.priorities_quant = [low_to_high[priorities[0]], low_to_high[priorities[1]], low_to_high[priorities[2]]]
        self.log_probs = []

class Dialog:
    def __init__(self, agent1, agent2):
        self.agents = [agent1, agent2]
        self.dialog_history = []
        self.num_rounds = 10

    def selfplay(self):
        random.shuffle(self.agents)
        flag = False
        return_val = None 
        starter = True
        for a in range(self.num_rounds):
            for agent in self.agents:
                prev_convo = self.dialog_history[-4:]
                convo_str = ""
                if len(prev_convo) > 0:
                    for i in prev_convo:
                        you_or_them = "YOU: " if list(i.keys())[0] == agent.id else "THEM: "
                        convo_str += f"{you_or_them}: {list(i.values())[0]} "
                convo_str += "YOU: "

                self.dialog_history.append({agent.id:agent.respond(convo_str, starter)})
                starter = False
                if "Accept-Deal" in list(self.dialog_history[-1].values())[0]:
                    flag = True
                    return_val = self.dialog_history
                    break
                if "Walk-Away" in list(self.dialog_history[-1].values())[0]:
                    flag = True
                    return_val = "Walk-Away"
                    break
            if flag:
                break
        
        return return_val if flag else None
    
    def print_dialog(self):
        for line in self.dialog_history:
            print(line)

num_epochs = 10
possible_priorities = list(itertools.permutations(["Low", "Medium", "High"], 3))
batch_size = 6

def get_reward(dialog, agent):
    try:
        if dialog is None:
            print("REWARD 0")
            return 0
        item_value = {2: 5, 1: 4, 0: 3}
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
                final_score = 0
                for a in range(len(agent_offer)):
                    final_score += item_value[agent.priorities_quant[a]] * int(agent_offer[a])
                print("VALID OFFER, Final Score:", final_score)
                return final_score
        else:
            print("Walk-Away")
            return 6
    except:
        print("ERROR")
        print(dialog)
        return None

def reinforce_loop():
    reinforce_agent = FlanAgent("reinforce_agent", "flan_t5-small-casino/checkpoint-14120")
    partner_agent = FlanAgent("partner_agent", "flan_t5-small-casino/checkpoint-14120")
    optimizer = torch.optim.AdamW(reinforce_agent.model.parameters(), lr=1e-4)
    epoch_reward = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        batch_log_probs = []
        batch_rewards = []
        total_loss = 0
        
        for prio in possible_priorities:
            for partner_prio in possible_priorities:
                for _ in range(batch_size):
                    reinforce_agent.initialize(prio)
                    partner_agent.initialize(partner_prio)
                    dialog = Dialog(reinforce_agent, partner_agent)

                    selfplay_result = dialog.selfplay()

                    reward = get_reward(selfplay_result, reinforce_agent)
                    if reward is None:
                        continue

                    # Ensure log_probs tensor requires grad
                    log_probs = torch.tensor(reinforce_agent.log_probs, dtype=torch.float32)
                    if log_probs.requires_grad == False:
                        log_probs.requires_grad_(True)

                    rewards = torch.tensor([reward] * len(log_probs), dtype=torch.float32)

                    # Compute the loss
                    loss = -torch.sum(log_probs * rewards)
                    total_loss += loss.item()

                    # Accumulate gradients
                    loss.backward()

                # Perform optimizer step and clear gradients
                torch.nn.utils.clip_grad_norm_(reinforce_agent.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / (len(possible_priorities) ** 2 * batch_size)
        print(f"Avg Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        temp = Dialog(reinforce_agent, partner_agent)
        temp.selfplay()
        temp.print_dialog()

    reinforce_agent.model.save_pretrained("rl_trained", from_pt=True)

reinforce_loop()
