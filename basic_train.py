# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
# import datasets
import pandas as pd
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

df = pd.read_parquet("hf://datasets/kchawla123/casino/data/train-00000-of-00001.parquet")
# print(df.head())

# training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
# metric = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

CONTEXT_SIZE = 4
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
        start_index = max(0, i - CONTEXT_SIZE)
        values = arr[start_index:i+1]
        result.append(values)
    
    results += result
    dialogs.append(dialog)

# print(dialogs[0])   
you_vs_them = {"mturk_agent_1":"YOU", "mturk_agent_2":"THEM"}
them_vs_you = {"mturk_agent_1":"THEM", "mturk_agent_2":"YOU"}
pre_processed_results = []
for result in results:
    base_str = ""
    base_str_inverted = ""
    # print(result)
    for r in result:
        k = list(r.keys())[0]
        v = r[k]
        base_str += f"{you_vs_them[k]}: {v} "
        base_str_inverted += f"{them_vs_you[k]}: {v} "
    pre_processed_results.append(base_str)

print(pre_processed_results[:5])


print(tokenizer.chat_template)
print(tokenizer.eos_token)

#current tasks
#Either fit to DialoGPT or T5(current line is response, previous lines are context)
#tokenize the dialog
#train the model









# print(results[0])
# print(dialogs)
# with open("data.txt", "w") as f:
#     f.write(str(dialogs[0]))
print(tokenizer.chat_template)
print(tokenizer.eos_token)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# print(df['chat_logs'][0])
