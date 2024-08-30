from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import random
import torch
from torch.nn.functional import log_softmax
import itertools
import numpy as np
import argparse
import logging
from tqdm import tqdm
import torch.nn.functional as F
import datetime

opponent_generation_params = {
    "max_new_tokens": 100,
    "do_sample": True, 
    "top_k": 50,
    "top_p": 0.92,
    "return_dict_in_generate": True,
    "output_logits": True
}

POSSIBLE_PRIORITIES = list(itertools.product(list(itertools.permutations(["Low", "Medium", "High"], 3)), repeat=2))

debug = False

class FlanAgent:
    def __init__(self, id, model_dir):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.priorities = "Priorities: Low Firewood Medium Water High Food "
        self.priorities_quant = [0, 1, 2]
        self.id = id
        self.log_probs = []

    def respond_generate(self, text, starter):
        if debug:
            print(text)
        inputs = self.tokenizer(["Continue writing the following text.\n\n" + self.priorities + text], return_tensors="pt")
        outputs = self.model.generate(**inputs, **opponent_generation_params)
        return self.tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
    
    def respond_forward_pass(self, text, starter):
        # USE THE MODEL ITSELF INSTEAD OF model.generate, the issue is that the tensors model.generate returns can not be used to update the optimizer

        if debug:
            print(text)
        inputs = self.tokenizer(["Continue writing the following text.\n\n" + self.priorities + text], return_tensors="pt")
        
        # Initialize variables for storing generated sequence and logits
        generated_sequence = inputs["input_ids"]
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]], device=generated_sequence.device)  # Start with the PAD token
        log_probs = []
        actions = []
        for _ in range(opponent_generation_params["max_new_tokens"]):
            outputs = self.model(input_ids=generated_sequence, decoder_input_ids=decoder_input_ids, return_dict=True)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            log_prob = log_softmax(next_token_logits, dim=-1)
            chosen_log_prob = log_prob[0, next_token_id]
            log_probs.append(chosen_log_prob)
            actions.append(next_token_id)


            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(0)], dim=-1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode the generated sequence into text
        final_text = self.tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        self.log_probs.extend(log_probs)


        # print(final_text)
        return final_text


    def initialize(self, priorities):
        low_to_high = {"Low": 0, "Medium": 1, "High": 2}
        self.priorities = f"Priorities: {priorities[0]} Firewood {priorities[1]} Water {priorities[2]} Food "
        self.priorities_quant = [low_to_high[priorities[0]], low_to_high[priorities[1]], low_to_high[priorities[2]]]
        self.log_probs = []

class Dialog:
    def __init__(self, agent1, agent2, args):
        self.agents = [agent1, agent2]
        self.dialog_history = []
        self.args = args

    def selfplay(self):
        random.shuffle(self.agents)
        flag = False
        return_val = None 
        starter = True
        for a in range(self.args.num_rounds):
            for agent in self.agents:
                prev_convo = self.dialog_history[-4:]
                convo_str = ""
                if len(prev_convo) > 0:
                    for i in prev_convo:
                        you_or_them = "YOU: " if list(i.keys())[0] == agent.id else "THEM: "
                        convo_str += f"{you_or_them}: {list(i.values())[0]} "
                convo_str += "YOU: "
                if agent.id == "reinforce_agent":
                    response = agent.respond_forward_pass(convo_str, starter)
                else:
                    response = agent.respond_generate(convo_str, starter) 
                self.dialog_history.append({agent.id:response})
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


        # self.print_dialog()


        return return_val if flag else None
    
    def print_dialog(self):
        for line in self.dialog_history:
            logging.info(line)
            print(line)


class Reinforcer:
    def get_reward(dialog, our_agent, partner_agent, utility):
        try:
            if dialog is None or dialog == "Walk-Away":
                logging.debug("Walkaway reward 12")
                return 18 if utility == "selfish" else 12
            item_value = {2: 5, 1: 4, 0: 3}
            if dialog != "Walk-Away":
                final_offers = dialog[-2:]
                agent_offer = final_offers[0] if list(final_offers[0].keys())[0] == our_agent.id else final_offers[1]
                other_offer = final_offers[0] if list(final_offers[0].keys())[0] != our_agent.id else final_offers[1]
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
                        final_score += item_value[our_agent.priorities_quant[a]] * int(agent_offer[a])
                    partner_final_score = 0
                    for a in range(len(other_offer)):
                        partner_final_score += item_value[partner_agent.priorities_quant[a]] * int(other_offer[a])
                    if utility != "selfish":
                        #utility function
                        reward = final_score - (0.75*max(0, partner_final_score - final_score)) - (0.75* max(0, final_score - partner_final_score))
                    else:
                        reward = final_score

                    logging.debug("VALID OFFER, Final Score:", reward)
                    return reward
        except Exception as e:
            logging.error("ERROR:", e)
            logging.info(dialog)
            return None

    def reinforce_loop(args):
        reinforce_agent = FlanAgent("reinforce_agent", args.model_dir)
        partner_agent = FlanAgent("partner_agent", args.model_dir)


        for name, param in reinforce_agent.model.named_parameters():
            if "decoder.block" in name and any(layer in name for layer in [".7", ".6", ".5", ".4", ".3"]):
                param.requires_grad = True
            elif "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False                   


        for name, param in reinforce_agent.model.named_parameters():
            if param.requires_grad:
                print(f"Fine-tuning: {name}")

        optimizer = torch.optim.SGD(reinforce_agent.model.parameters(), lr=1e-4, momentum=0.9)
        epoch_reward = []
        logging.basicConfig(filename=args.log_file, level=logging.INFO if args.logging_level == "INFO" else logging.DEBUG)
        epoch_tqdm = tqdm(range(args.num_epochs), desc="Epochs")
        prio_tqdm = tqdm(POSSIBLE_PRIORITIES, desc="Priorities")

        prio_averages = {}

        # torch.autograd.set_detect_anomaly(True)

        reinforce_agent.model.train()
        for epoch in range(args.num_epochs):
            logging.info(f"Epoch {epoch + 1}/{args.num_epochs}")            
            total_loss = 0
            epoch_rewards = 0
            
            for prio, partner_prio in POSSIBLE_PRIORITIES:
                # initial_params = {name: param.clone() for name, param in reinforce_agent.model.named_parameters()}
                batch_info = []
                batch_reward = 0
                
                for _ in range(args.batch_size):              
                    reinforce_agent.initialize(prio)
                    partner_agent.initialize(partner_prio)
                    dialog = Dialog(reinforce_agent, partner_agent, args)

                    selfplay_result = dialog.selfplay()

                    reward = Reinforcer.get_reward(selfplay_result, reinforce_agent, partner_agent, args.utility)
                    if reward is not None:
                        
                        batch_reward += reward
                        
                        
                        reward = reward/36.00 # normalize against max reward of 36
                        batch_info.append((reinforce_agent.log_probs, reward))

                loss = 0
                if (tuple(prio),tuple(partner_prio)) not in prio_averages:
                    prio_averages[(tuple(prio),tuple(partner_prio))] = 0
                prio_averages[(tuple(prio),tuple(partner_prio))] += (batch_reward/float(args.batch_size)) / float(epoch + 1)       
                

                #cant do multiple epochs for PPO, need to do it in one epoch
                for log_probs, reward in batch_info:
                    advantage = reward - prio_averages[(tuple(prio),tuple(partner_prio))]
                    for log_prob in log_probs:
                        loss += -log_prob * advantage

                if loss != 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(reinforce_agent.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                # for _ in range(args.ppo_epochs):  # Multiple epochs for PPO
                #     for log_probs, reward in batch_info:
                #         optimizer.zero_grad()
                #         loss = 0
                #         advantage = reward - prio_averages[(tuple(prio),tuple(partner_prio))]
                #         for log_prob in log_probs:
                #             log_prob = log_prob.detach()  # Detach to avoid in-place modification issues
                #             ratio = torch.exp(log_prob - log_prob.detach())
                #             surr1 = ratio * advantage
                #             surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantage
                #             loss = loss + (-torch.min(surr1, surr2).mean())
                #         if loss != 0:
                #             loss.backward(retain_graph=True)  # Prevent in-place operation issues
                #             torch.nn.utils.clip_grad_norm_(reinforce_agent.model.parameters(), 1.0)
                #             optimizer.step()
                #             optimizer.zero_grad()  # Clear gradients after the step
                #             total_loss += loss.item()
                #         else:
                #             logging.warning("Loss is 0, skipping step")
                
                    

                epoch_rewards += batch_reward/float(args.batch_size)
                logging.info(f"Reward: {batch_reward/float(args.batch_size)}")
                prio_tqdm.update(1)
                

            avg_loss = total_loss / (args.batch_size * args.ppo_epochs * (len(POSSIBLE_PRIORITIES) ** 2))
            logging.info(f"Avg Loss: {avg_loss:.4f}")
            logging.info(f"Total Reward: {epoch_rewards}")
            logging.info(f"Avg Reward: {epoch_rewards /(36.0)}")

            prio_tqdm.reset()
            epoch_tqdm.update(1)

            # Evaluate after each epoch
            temp = Dialog(reinforce_agent, partner_agent, args)
            temp.selfplay()
            temp.print_dialog()

        reinforce_agent.model.save_pretrained("rl_trained", from_pt=True)

        prio_tqdm.close()
        epoch_tqdm.close()


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--num_epochs", type=int, default=15)
arg_parser.add_argument("--batch_size", type=int, default=4)
arg_parser.add_argument("--debug", action="store_true")
arg_parser.add_argument("--model_dir", type=str, default="flan_t5-small-casino/checkpoint-14120")
arg_parser.add_argument("--output_dir", type=str, default="rl_trained")
arg_parser.add_argument("--num_rounds", type=int, default=10)
arg_parser.add_argument("--utility", type=str, default="selfish")
arg_parser.add_argument("--logging_level", type=str, default="INFO")
arg_parser.add_argument("--log_file", type=str, default=f"logs/reinforce{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log")
arg_parser.add_argument("--ppo_epochs", type=int, default=4)
arg_parser.add_argument("--ppo_clip", type=float, default=0.2)
parsed_args = arg_parser.parse_args()



Reinforcer.reinforce_loop(parsed_args)