from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import pandas as pd
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import datasets
from transformers import DataCollatorForSeq2Seq
import random
generation_params = {
    "max_length": 600,
    "no_repeat_ngram_size": 1,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.7,
    "num_return_sequences": 1,
    "repetition_penalty": 1.3,
    "return_dict_in_generate":True,
    "output_scores" : True


}
debug = False
class FlanAgent:
    def __init__(self, id, model_dir):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir,local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        self.priorities = "Priorities: Low Firewood Medium Water High Food  "
        self.priorities_quant = [0,1,2]
        self.id = id
        self.log_probs = []

    def respond(self, text):
        if debug:
            print(text)
        inputs = self.tokenizer(["Continue writing the following text.\n\n"+ self.priorities + text], return_tensors="pt")
        reply_ids = self.model.generate(**inputs, **generation_params)
        print(len(reply_ids['scores']))
        #process log_probs here
        log_probs = np.array(reply_ids['scores'])
        log_probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
        log_probs = list(np.log(log_probs))
        self.log_probs.extend(log_probs)
        return self.tokenizer.decode(reply_ids['sequences'][0], skip_special_tokens=True)
    
    def setPriorities(self, priorities):
        low_2_high = {"Low":0, "Medium":1, "High":2}
        self.priorities = f"Priorities: {priorities[0]} Firewood {priorities[1]} Water {priorities[2]} Food  "
        self.priorities_quant = [low_2_high[priorities[0]], low_2_high[priorities[1]], low_2_high[priorities[2]]]
    
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
        
        return return_val
    
    def print_dialog(self):
        for line in self.dialog_history:
            print(line)

Dialog([FlanAgent("agent1","flan_t5-small-casino/checkpoint-14120"), FlanAgent("agent2","flan_t5-small-casino/checkpoint-14120")]).selfplay()

        