import pandas as pd
from pygments.lexers import PythonLexer
import csv

FOLDER_PATH = "CSCI_420/CSCI420-Assignment-2"
lexer = PythonLexer()
train_corpus = pd.read_csv(f'{FOLDER_PATH}/Archive/ft_train.csv')
test_corpus = pd.read_csv(f'{FOLDER_PATH}/Archive/ft_test.csv')
valid_corpus = pd.read_csv(f'{FOLDER_PATH}/Archive/ft_valid.csv')

def join(tokens):
    """Custom join function to merge tokens after using Pygments tokenizer.

    Args:
        tokens: A list containing a tokenized method

    Returns:
        A string representing the tokenized method
    """
    joined = ""
    for token in tokens:
        if token == "\n" or "    " in token:
            joined += token
        else:
            joined += token + " "
    return joined

def mask(input_data, output_file):
    with open(f"{FOLDER_PATH}/data/{output_file}", mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Masked Method", "Target Code"])

        for i in range(0, len(input_data["cleaned_method"])):
            method = input_data["cleaned_method"][i]
            target = input_data["target_block"][i]
            method_tokens = [t[1] for t in lexer.get_tokens(method) if t[1] != ' ']
            target_tokens = [t[1] for t in lexer.get_tokens(target) if t[1] != ' ']
            target = " ".join(target_tokens)
            method = join(method_tokens).replace("    ", "<TAB> ").replace(target, "<MASK> ", 1).replace("\n", "")
            csv_writer.writerow([method, target])

print("Masking training set")
mask(train_corpus, "train_set.csv")
print("Masking test set")
mask(test_corpus, "test_set.csv")
print("Masking valid set")
mask(valid_corpus, "valid_set.csv")