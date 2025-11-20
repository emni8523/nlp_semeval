import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset/training_data_processed.csv')


df["q_len"] = df["question"].str.split().str.len()
df["a_len"] = df["answer"].str.split().str.len()

print(df[["q_len", "a_len"]].describe())



df["q_len"].hist(bins=30)
plt.title("Question Length Distribution")
plt.show()

df["a_len"].hist(bins=30)
plt.title("Answer Length Distribution")
plt.show()