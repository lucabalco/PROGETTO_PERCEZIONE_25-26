import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 


df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df.rename(columns={df.columns[-1]: 'Obesity_Level'}, inplace=True)

df['BMI'] = df['Weight'] / (df['Height'] ** 2)

target_classes_1 = ['Normal_Weight', 'Insufficient_Weight']
df['Binary_Target'] = np.where(df['Obesity_Level'].isin(target_classes_1), 1, 0)

counts = df['Binary_Target'].value_counts().sort_index()
indices = counts.index
heights = counts.values


plt.figure(figsize=(9, 5), facecolor="#fff7eb")
plt.bar(indices, heights, width=0.4, color=["#B65337","#FFbc7e"], align='center',edgecolor=["#b65337","#FFbc7e"])
plt.xlabel('')
plt.xticks([0, 1], ['Sovrappeso/Obeso', 'Normopeso/Sottopeso'], 
        rotation=0, 
        ha='center', 
        fontsize=15,
        fontname="Prata", 
        color="#ff914d",
    ) 
plt.ylabel('') 

ax = plt.gca() 
ax.set_xlim(-1, 2)
ax.set_facecolor("#fff7eb") 
ax.tick_params(axis='x', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.grid(False) 
ax.set_yticks([])    


for i, h in zip(indices, heights):
    ax.text(i, h+50, f'{int(h)}', ha="center", fontsize=24, color="#FF914D", fontname="Prata", weight='bold' )
plt.show()


plt.figure(figsize=(3, 5), facecolor="#fff7eb") 

sns.boxplot(
    df['Age'],
    notch=True,
    boxprops={
        'alpha': 1.0,
        'facecolor': "#b65337", 
        'edgecolor': "#b65337"  
    },
    flierprops={
        "marker":"x",
        'color' : '#ff914d'
    },
    medianprops={
        'color' : "#000000",
        'linewidth' : '1'
    }
)
plt.title('Età', fontsize=24, fontname="Prata", color="#b65337", weight= "bold")
plt.xlabel('')
plt.ylabel('Frequenza', labelpad=10, fontname="Prata", color="#b65337", fontsize=16, weight='bold')
plt.yticks(color="#b65337", fontname="Prata", fontsize=12, weight='bold')
cx = plt.gca() 
cx.spines['top'].set_visible(False)
cx.spines['right'].set_visible(False)
cx.spines['left'].set_visible(False)
cx.spines['bottom'].set_visible(False)
cx.set_facecolor("#fff7eb")
cx.tick_params(axis='x', length=0)

plt.tight_layout()
plt.show()

plt.figure(figsize=(3, 5), facecolor="#fff7eb") 

sns.boxplot(
    df['BMI'],
    notch=True,
    boxprops={
        'alpha': 1.0,
        'facecolor': "#ff914d", 
        'edgecolor': "#ff914d"  
    },
    medianprops={
        'color' : "#000000",
        'linewidth' : '1'
    }
)
plt.title('BMI', fontsize=24, fontname="Prata", color="#b65337", weight= "bold")
plt.xlabel('')
plt.ylabel('Frequenza', labelpad=10, fontname="Prata", color="#b65337", fontsize=16, weight='bold')
plt.yticks(color="#b65337", fontname="Prata", fontsize=12, weight='bold')
cx = plt.gca() 
cx.spines['top'].set_visible(False)
cx.spines['right'].set_visible(False)
cx.spines['left'].set_visible(False)
cx.spines['bottom'].set_visible(False)
cx.set_facecolor("#fff7eb")
cx.tick_params(axis='x', length=0)

plt.tight_layout()
plt.show()
