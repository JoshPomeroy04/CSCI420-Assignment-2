import pandas as pd
from pygments.lexers import PythonLexer

FOLDER_PATH = "CSCI_420/CSCI420-Assignment-2"
lexer = PythonLexer()
csv_corpus = pd.read_csv(f'{FOLDER_PATH}/Archive/ft_test.csv')

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

target = csv_corpus["target_block"][0]
method = csv_corpus["cleaned_method"][0]
method_tokens = [t[1] for t in lexer.get_tokens(method) if t[1] != ' ']
target_tokens = [t[1] for t in lexer.get_tokens(target) if t[1] != ' ']

target = " ".join(target_tokens)
method = join(method_tokens).replace("    ", "<TAB> ").replace(target, "<MASK> ", 1)

print(method.replace("\n", ""))
