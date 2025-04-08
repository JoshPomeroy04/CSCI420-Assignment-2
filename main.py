from datasets import load_dataset
from transformers import T5ForConditionalGeneration
from transformers import RobertaTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

FOLDER_PATH = "CSCI_420/CSCI420-Assignment-2"
train_file = f"{FOLDER_PATH}/data/train_set.csv"
test_file = f"{FOLDER_PATH}/data/test_set.csv"
valid_file = f"{FOLDER_PATH}/data/valid_set.csv"
print("Loading datasets\n")
dataset = load_dataset("csv", data_files={"train": train_file, "test": test_file, "validation": valid_file})

model_checkpoint = "Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["<MASK>"])
tokenizer.add_tokens(["<TAB>"])

print("Resizing embeddings\n")
model.resize_token_embeddings(len(tokenizer))

def preprocess_function(examples):
    inputs = examples["Masked Method"]
    targets = examples["Target Code"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=f"{FOLDER_PATH}/codet5-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=7,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=100,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Training Model\n")
trainer.train()

metrics = trainer.evaluate(tokenized_datasets["test"])
print("Test Evaluation Metrics:", metrics)
