import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

normopeso_classi = ["Normal_Weight", "Insufficient_Weight"]
df["Peso_binario"] = df["NObeyesdad"].apply(
    lambda x: "Normopeso/Sottopeso" if x in normopeso_classi else "Sovrappeso/Obeso"
)

binary_features = [col for col in df.columns if df[col].nunique() == 2 and col != "Peso_binario"]

titoli = ['Genere', 'Famiglia con storia di obesità', 'Consumazione di cibo altamente calorici', 'Fumatore', 'Controllo delle calorie giornaliero']
xticks= [['Donna','Uomo'],['No','Sì'],['No','Sì'],['No','Sì'],['No','Sì']]

for i, feature in enumerate(binary_features):
    counts = pd.crosstab(df[feature], df["Peso_binario"])
    
    x = range(len(counts))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(5, 6), facecolor='#fff7eb')
    
    bars1 = ax.bar([p - width/2 for p in x], counts["Sovrappeso/Obeso"], width=width, color="#B65337", label="Sovrappeso/Obeso")
    bars2 = ax.bar([p + width/2 for p in x], counts["Normopeso/Sottopeso"], width=width, color="#FFbc7e", label="Normopeso/Sottopeso")
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.3,
            int(height),
            ha='center',
            va='bottom',
            fontsize=16,
            fontname='Prata',
            color='#b65337'
        )
    ax.set_facecolor('#fff7eb')
    ax.set_xticks(x)
    ax.set_xticklabels(xticks[i], color='#b65337', fontname='Prata', fontsize=16)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    
    ax.tick_params(axis='x', which='both', length=0)
    ax.minorticks_off()
    ax.set_ylim(0, max(counts.max())*1.15)
    
    plt.title(titoli[i], fontsize=18, loc='center', fontname='Prata',color="#B65337" )
    
    plt.tight_layout()
    plt.show()
