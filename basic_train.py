# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# import datasets
import pandas as pd
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

df = pd.read_parquet("hf://datasets/kchawla123/casino/data/train-00000-of-00001.parquet")
# print(df.head())

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print(df["chat_logs"][:][:]['text'])
small_train_dataset = tokenizer(df['chat_logs']['text'])[:900]
small_eval_dataset = tokenizer(df['chat_logs']['text'])[900:]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# print(df['chat_logs'][0])
