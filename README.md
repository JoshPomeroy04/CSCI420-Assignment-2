# CSCI420-Assignment-2

# **1. Introduction**  
This project explores **fine tuning models**. In this project, the codet5-small model is fine tuned in order to be able to predict if statements. A Python method with a masked if statement is fed into the model and as output, the model generates an if statement that it thinks belongs there.

# **2. Getting Started**  

This project is implemented in **Python 3.10.12** 64-bit. It was developed and tested on **Ubuntu 22.04.3 LTS**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/JoshPomeroy04/CSCI420-Assignment-2.git
```
(2) Navigate into the repository



Ensure you set the FOLDER_PATH variable to be the correct folder path of the workspace. This variable is easily found at the top of each .py file

## **2.2 Install Packages**

Install the required dependencies: \
python -m pip install tree-sitter==0.23 \
python -m pip install tree-sitter-python \
python -m pip install sacrebleu \
python -m pip install pygments \
python -m pip install csv \
python -m pip install pandas \
python -m pip install datasets \
python -m pip install transformers \
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

## **2.3 Train Model**

The model can be trained by simply running main.py The number of epochs and save directory can be changed in main.py
If you have your own data sets that have not been preprocessed yet, you can run mask.py to mask and flatten the data sets.
generate_predictions.py generates predicted statements by a specified model. 

If you want to go through the process of data processing, the original data can be found here: https://piazza.com/class_profile/get_resource/m5yv2dhoopb363/m8rm0p0lved54j \
Models that I have trained can be found here: https://drive.google.com/file/d/1VMbExH4zCX1FwUZ0uhcpnf6p-Q9yHHoe/view?usp=sharing \

## **2.4 Evaluate**

The given eval files are predictions generated from 4 different models in addition to the target statements. To change which file you run the evaluation on simply change the number in predictions[num].txt. The following is what you can put in the command line to run the evaluation.

cd CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU && python calc_code_bleu.py --refs CSCI420-Assignment-2/eval/targets.txt --hyp CSCI420-Assignment-2/eval/predictions9.txt --lang python --params 0.25,0.25,0.25,0.25

Additionally, results.py compiles the information into a csv file which includes: the masked input method, the expected if statement, the generated if statement, the CodeBLEU score, the BLEU score, and if the expected and generated are an exact match.


