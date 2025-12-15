import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. Caricamento dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# 2. Creazione target binario
normopeso_classi = ["Insufficient_Weight", "Normal_Weight"]
df["Peso_binario"] = df["NObeyesdad"].apply(lambda x: 0 if x in normopeso_classi else 1)

# 3. Rimozione target originale
df = df.drop(columns=["NObeyesdad"])

# 4. Label encoding delle categoriche
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# 5. Matrice di correlazione
corr = df_encoded.corr()

# 6. Heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(
    corr,
    cmap="coolwarm",
    linewidths=0.4,
    annot=True,
    fmt=".2g", 
    annot_kws={"size": 12},
    cbar_kws={"shrink": 0.6}
)


plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
