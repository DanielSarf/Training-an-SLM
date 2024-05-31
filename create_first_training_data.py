import numpy as np
import os
import json
import pandas as pd
import fastparquet
import random
from transformers import AutoTokenizer

# Load locally downloaded datasets:

# https://huggingface.co/datasets/pankajmathur/WizardLM_Orca
with open(os.path.join(os.path.dirname(__file__), "Data\\Datasets\\wizardlm_orca.json"), 'r', encoding='utf-8') as f:
    WizardLLMOrca = f.read()
    WizardLLMOrca = json.loads(WizardLLMOrca)

for i in range(len(WizardLLMOrca)):
    text = ""
    text += "<|user|>:\n" + WizardLLMOrca[i]["instruction"]
    text += "\n<|assistant|>:\n" + WizardLLMOrca[i]["output"]
    WizardLLMOrca[i] = text

# https://huggingface.co/datasets/garage-bAInd/Open-Platypus
OpenPlatypusDF = pd.read_parquet(os.path.join(os.path.dirname(__file__), "Data\\Datasets\\train-00000-of-00001-4fe2df04669d1669.parquet"), engine='fastparquet')
OpenPlatypus = []

for i in range(len(OpenPlatypusDF)):
    text = ""
    text += "<|user|>:\n" + OpenPlatypusDF["instruction"][i]
    text += "\n<|assistant|>:\n" + OpenPlatypusDF["output"][i]
    OpenPlatypus.append(text)

del OpenPlatypusDF

# https://huggingface.co/datasets/databricks/databricks-dolly-15k
DatabricksDolly15KDF = pd.read_json(path_or_buf = os.path.join(os.path.dirname(__file__), "Data\\Datasets\\databricks-dolly-15k.jsonl"), lines=True)
DatabricksDolly15K = []

for i in range(len(DatabricksDolly15KDF)):
    text = ""
    text += "<|user|>:\n" + DatabricksDolly15KDF["instruction"][i]
    text += "\nContext: " + DatabricksDolly15KDF["context"][i] if DatabricksDolly15KDF["context"][i] != "" else ""
    text += "\n<|assistant|>:\n" + DatabricksDolly15KDF["response"][i]
    DatabricksDolly15K.append(text)

del DatabricksDolly15KDF

# https://huggingface.co/datasets/Open-Orca/OpenOrca
OpenOrcaGPT3_5DF = pd.read_parquet(os.path.join(os.path.dirname(__file__), "Data\\Datasets\\3_5M-GPT3_5-Augmented.parquet"), engine='fastparquet')
OpenOrcaGPT3_5 = []

for i in range(len(OpenOrcaGPT3_5DF)):
    text = ""
    text += "<|user|>:\n" + OpenOrcaGPT3_5DF["question"][i]
    text += "\n<|assistant|>:\n" + OpenOrcaGPT3_5DF["response"][i]
    OpenOrcaGPT3_5.append(text)

del OpenOrcaGPT3_5DF

OpenOrcaGPT4DF = pd.read_parquet(os.path.join(os.path.dirname(__file__), "Data\\Datasets\\1M-GPT4-Augmented.parquet"), engine='fastparquet')
OpenOrcaGPT4 = []

for i in range(len(OpenOrcaGPT4DF)):
    text = ""
    text += "<|user|>:\n" + OpenOrcaGPT4DF["question"][i]
    text += "\n<|assistant|>:\n" + OpenOrcaGPT4DF["response"][i]
    OpenOrcaGPT4.append(text)

del OpenOrcaGPT4DF

# Combine datasets and remove duplicates 
combinedDataset = list(set(WizardLLMOrca + OpenPlatypus + DatabricksDolly15K + OpenOrcaGPT3_5 + OpenOrcaGPT4))

del WizardLLMOrca, OpenPlatypus, DatabricksDolly15K, OpenOrcaGPT3_5, OpenOrcaGPT4

# Shuffle the dataset
combinedDataset = random.sample(combinedDataset, len(combinedDataset))

#Split dataset into train and validation data
n = len(combinedDataset)
trainData = combinedDataset[:int(n * 0.99)]
valData = combinedDataset[int(n * 0.99):]

del combinedDataset

# Tokenize all dataset entries, append EOT token after each of them
tokenizer = AutoTokenizer.from_pretrained("gpt2")
endOfTextToken = tokenizer("<|endoftext|>").input_ids[0]

maxTokenLength = 2048

#Init array of 0s which may be bigger than needed, but this increases efficency of putting the tokens into the IDs array
totalTrainEntries = len(trainData)
trainIDs = np.zeros(totalTrainEntries * maxTokenLength, dtype = np.uint16)

trainIndex = 0
currentTrainToken = 0

for entry in trainData:
    #I did maxTokenLength - 1, because EOT token needs to be added aswell
    tokenizedEntry = tokenizer(entry, truncation = True, max_length = maxTokenLength - 1).input_ids
    numOfTokensInEntry = len(tokenizedEntry)

    trainIDs[currentTrainToken:currentTrainToken + numOfTokensInEntry] = tokenizedEntry[0:numOfTokensInEntry]
    currentTrainToken = currentTrainToken + numOfTokensInEntry
    
    trainIDs[currentTrainToken] = endOfTextToken
    currentTrainToken += 1

    trainIndex += 1
    if trainIndex % 1000 == 0:
        print(f"{trainIndex} of {totalTrainEntries} entries from the training data have been encoded - {format(100 * trainIndex / totalTrainEntries, '.2f')}% done.")

trainIDs = trainIDs[:currentTrainToken]

print(f"All {totalTrainEntries} entries of the training data have been encoded. Total train tokens = {currentTrainToken}")

del trainData

trainIDs.tofile(os.path.join(os.path.dirname(__file__), "Data\\Training and Validation Data\\First Training Data\\Train.bin"))

del trainIDs

totalValEntries = len(valData)
valIDs = np.zeros(totalValEntries * maxTokenLength, dtype = np.uint16)

valIndex = 0
currentValToken = 0

for entry in valData:
    tokenizedEntry = tokenizer(entry, truncation = True, max_length = maxTokenLength - 1).input_ids
    numOfTokensInEntry = len(tokenizedEntry)

    valIDs[currentValToken:currentValToken + numOfTokensInEntry] = tokenizedEntry[0:numOfTokensInEntry]
    currentValToken = currentValToken + numOfTokensInEntry
    valIDs[currentValToken] = endOfTextToken
    currentValToken += 1
    
    valIndex += 1
    if valIndex % 1000 == 0:
        print(f"{valIndex} of {totalValEntries} entries from the validation data have been encoded - {format(100 * valIndex / totalValEntries, '.2f')}% done.")

valIDs = valIDs[:currentValToken]

print(f"All {totalValEntries} entries of the validation data have been encoded. Total validation tokens = {currentValToken}")

del valData

valIDs.tofile(os.path.join(os.path.dirname(__file__), "Data\\Training and Validation Data\\First Training Data\\Val.bin"))

del valIDs