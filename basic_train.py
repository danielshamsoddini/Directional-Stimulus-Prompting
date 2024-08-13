# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# import datasets
import pandas as pd
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np


# tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
# model = AutoModelForCausalLM.from_pretrained("google-t5/t5-base")

df = pd.read_parquet("hf://datasets/kchawla123/casino/data/train-00000-of-00001.parquet")
# print(df.head())

# training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
# metric = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

CONTEXT_SIZE = 5
other_id = {"mturk_agent_1":"mturk_agent_2", "mturk_agent_2":"mturk_agent_1"}
dialogs = []
results = []
for _,a in df.iterrows():
    dialog = []
    priorities = [a['participant_info']['mturk_agent_1']["value2issue"], a['participant_info']['mturk_agent_2']["value2issue"]]
    final_offer = {}
    for b in a['chat_logs']:
        # print(b)
        
        if b['text'] == "Submit-Deal":
            # print(b)
            # final_offer[b['id']] = b['task_data']['issue2youget']
            # final_offer[other_id[b['id']]] = b['task_data']['issue2theyget']
            line_text =f"{b["text"]} {b['task_data']['issue2youget']['Firewood']} Firewood {b['task_data']['issue2youget']['Water']} Water {b['task_data']['issue2youget']['Food']} Food"
            dialog.append({b['id']:line_text})
        else:
            dialog.append({b['id']:b["text"]})
    result = []
    arr = dialog
    for i in range(len(arr)):
        if i < CONTEXT_SIZE:
            result.append(arr[:i])
        else:
            result.append(arr[i-CONTEXT_SIZE:i])
    
    results.append(result)
    dialogs.append(dialog)

print(dialogs[0])   
# print(results[0])
# print(dialogs)
# with open("data.txt", "w") as f:
#     f.write(str(dialogs[0]))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# print(df['chat_logs'][0])
