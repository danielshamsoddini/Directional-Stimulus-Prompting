from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import pandas as pd
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import datasets
from transformers import DataCollatorForSeq2Seq
import random
import torch
from torch.nn.functional import log_softmax
import itertools
generation_params = {
    "max_length": 600,
    # "no_repeat_ngram_size": 1,
    "do_sample": True,
    "top_k": 20,
    "top_p": 0.95,
    # "temperature": 0.7,
    # "num_return_sequences": 1,
    # "repetition_penalty": 1.3,
    "return_dict_in_generate": True,
    "output_scores": True
}
debug = False
class FlanAgent:
    def __init__(self, id, model_dir):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.priorities = "Priorities: Low Firewood Medium Water High Food  "
        self.priorities_quant = [0, 1, 2]
        self.id = id
        self.log_probs = []

    def respond(self, text):
        if debug:
            print(text)
        inputs = self.tokenizer(["Continue writing the following text.\n\n" + self.priorities + text], return_tensors="pt")
        outputs = self.model.generate(**inputs, **generation_params)

        # Process log probabilities
        # print(outputs['scores'])
        log_probs = []
        # print(outputs['scores'])    
        # enum_var = outputs['scores'][torch.isfinite(outputs['scores'])]
        # print(enum_var)
        # exit()
        for i, logits in enumerate(outputs['scores']):
            logits = torch.clamp(logits, -1000, 1000)  # Clamp logits to prevent overflow
            probs = log_softmax(logits, dim=-1)  # Get log-softmax over logits
            token_id = outputs['sequences'][0, i]  # Get token id
            log_probs.append(probs[0, token_id].item())  # Get log-prob of generated token

        self.log_probs.extend(log_probs)
        return self.tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)

    def setPriorities(self, priorities):
        low_to_high = {"Low": 0, "Medium": 1, "High": 2}
        self.priorities = f"Priorities: {priorities[0]} Firewood {priorities[1]} Water {priorities[2]} Food  "
        self.priorities_quant = [low_to_high[priorities[0]], low_to_high[priorities[1]], low_to_high[priorities[2]]]
        self.log_probs = []
class Dialog:
    def __init__(self, agents):
        self.agents = agents
        self.dialog_history = []
        self.num_rounds = 10

    def selfplay(self):
        # print(self.agents[0].model.parameters())
        random.shuffle(self.agents)
        flag = False
        return_val = None   
        for a in range(self.num_rounds):
            for agent in self.agents:
                prev_convo = self.dialog_history[-4:]
                convo_str = ""
                if len(prev_convo) > 0:
                    for i in prev_convo:
                        you_or_them = "YOU: " if list(i.keys())[0] == agent.id else "THEM: "
                        convo_str += f"{you_or_them}: {list(i.values())[0]} "
                convo_str += "YOU: "

                
                self.dialog_history.append({agent.id:agent.respond(convo_str)})
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
        
        if True:
            self.print_dialog()
            exit()
        
        return return_val
    
    def print_dialog(self):
        for line in self.dialog_history:
            print(line)


        