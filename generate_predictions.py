from datasets import load_dataset
from transformers import T5ForConditionalGeneration
from transformers import RobertaTokenizer

FOLDER_PATH = "CSCI_420/CSCI420-Assignment-2"
model_checkpoint = f"{FOLDER_PATH}/Model-5/checkpoint-125000"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
test_file = f"{FOLDER_PATH}/data/test_set.csv"
dataset = load_dataset("csv", data_files={"test": test_file})

# Create file containing target if statements
with open(f"{FOLDER_PATH}/eval/targets.txt", "w") as file:
    for statement in dataset["test"]["Target Code"]:
        file.write(statement)

# Create file containing generated predicted if statements
print("Generating predictions")

with open(f"{FOLDER_PATH}/eval/predictions.txt", "w") as file:
    count = 1
    for method in dataset["test"]["Masked Method"]:
        print(f"Generating Prediction {count} out of 5000")
        input_tensor = tokenizer(method, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**input_tensor, max_length=256)
        file.write(f"{(tokenizer.decode(outputs[0], skip_special_tokens=True)).strip()}\n")
        print(f"Preidiction {count} is: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        count += 1