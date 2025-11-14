from datasets import load_dataset
import pandas as pd
import os 


ds = load_dataset("ailsntua/QEvasion")

df_train = ds['train'].to_pandas()
df_test = ds['test'].to_pandas()

df_train.to_csv('dataset/training_data.csv', index=False)
df_test.to_csv('dataset/test_data.csv', index=False)


