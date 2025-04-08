from datasets import load_dataset
from transformers import T5ForConditionalGeneration
from transformers import RobertaTokenizer

FOLDER_PATH = "CSCI_420/CSCI420-Assignment-2"
model_checkpoint = f"{FOLDER_PATH}/Model-5/checkpoint-125000"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
test_file = f"{FOLDER_PATH}/data/test_set.csv"

print("Loading datasets\n")
dataset = load_dataset("csv", data_files={"test": test_file})

input_tensors = []
for method in dataset["test"]["Masked Method"]:
    input_tensor = tokenizer(method, return_tensors="pt", padding=True, truncation=True)
    input_tensors.append(input_tensor)


inputs = input_tensors[0]
outputs = model.generate(**inputs, max_length=256)
print("Generated if statement:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"Expected if statement:\n{dataset['test']['Target Code'][0]}")