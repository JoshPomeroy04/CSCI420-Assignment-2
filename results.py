import csv
from sacrebleu.metrics import BLEU
from datasets import load_dataset
from codebleu import calc_codebleu

FOLDER_PATH = "CSCI_420/CSCI420-Assignment-2"
test_file = f"{FOLDER_PATH}/data/test_set.csv"
prediction_file = f"{FOLDER_PATH}/eval/predictions9.txt"
dataset = load_dataset("csv", data_files={"test": test_file})

exact_count = 0
predictions = []
bleu = BLEU()

# Get all predictions
with open(f"{FOLDER_PATH}/eval/predictions9.txt", mode="r") as predfile:
        for line in predfile:
                predictions.append(line)

with open(f"{FOLDER_PATH}/testset-results.csv", mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input Function", "Expected If", "Predicted If", "CodeBLEU Score", "BLEU-4 Score", "Exact Match?"])
        for i in range(0, 5000):
                pred = predictions[i]
                fact = dataset["test"]["Target Code"][i]
                exact = "False"
                bleuscore = bleu.sentence_score(pred, [fact]).score
                if round(bleuscore, 0) == 100:
                        exact = "True"
                        exact_count += 1
                codebleuscore = calc_codebleu([fact], [pred], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                csv_writer.writerow([dataset["test"]["Masked Method"][i], fact, pred, 
                                     round(codebleuscore["codebleu"] * 100, 2) , round(bleuscore, 2), exact])

print(f"Number of exact matches: {exact_count}\n Percent exact matches: {(exact_count/5000)*100}")