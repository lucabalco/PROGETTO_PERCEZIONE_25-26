import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import os

# Fetch Dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features

colonne_richieste = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
dati_selezionati = X[colonne_richieste].copy() 

# Arrotondamento e Limitazione
dati_selezionati = np.ceil(dati_selezionati - 0.5).astype(int)
dati_selezionati = dati_selezionati.clip(upper=4)
sns.set_style("white") 

plot_details = {
    'FCVC': {
        'title': 'Frequenza Consumo di Vegetali', 
        'labels': {1: 'Mai', 2: 'A volte', 3: 'Sempre'}
    },
    'NCP': {
        'title': 'Numero di Pasti Principali', 
        'labels': {1: '1', 2: '2', 3: '3', 4: 'Più di 3'} 
    },
    'FAF': {
        'title': 'Frequenza Attività Fisica', 
        'labels': {0: 'Mai', 1: '1-2 volte/sett.', 2: '2-4 volte/sett.', 3: '4-5 volte/sett.'}
    },
    'CH2O':  {
        'title': 'Consumo d\'Acqua', 
        'labels': {1: 'Meno di 1L', 2: '1-2L', 3: 'Più di 2L'} 
    },
    'TUE': {
        'title': 'Uso di Dispositivi Tecnologici', 
        'labels': {0: '0-2 ore', 1: '3-5 ore', 2: 'Più di 5 ore'}
    }
}
 
for i, (col, details) in enumerate(plot_details.items()):     
    if col == "FAF" : 
        plt.figure(figsize=(8, 5), facecolor="#fff7eb")
    elif col =="FCVC":
        plt.figure(figsize=(6, 5), facecolor="#fff7eb")
    else:
        plt.figure(figsize=(5, 5), facecolor="#fff7eb")
    if (i+1)%2 ==1 :
        sns.histplot(
            data=dati_selezionati,
            x=col,
            bins=np.arange(0.5, 5.5, 1),
            discrete=True,
            color="#b65337" , 
            shrink=0.5,
            edgecolor="#b65337",
            alpha=1.0
        )
    else :
        sns.histplot(
            data=dati_selezionati,
            x=col,
            bins=np.arange(0.5, 5.5, 1),
            discrete=True,
            color="#FFbc7e" , 
            shrink=0.5,
            edgecolor="#FFbc7e",
            alpha=1.0
        )
    plt.title(f"{details['title']}", fontsize=24,fontname="Prata", y = 1.1, color="#b65337",weight= "bold")
    plt.ylabel('')
    
    ax = plt.gca()
    ax.set_facecolor("#fff7eb")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.get_yaxis().set_ticks([]) 
    
    tick_positions = list(details['labels'].keys())
    tick_labels = list(details['labels'].values())
    
    plt.xticks(tick_positions, tick_labels, ha='center',fontname="Prata", fontsize=15, color="#b65337")
    plt.xlabel('')
    plt.tight_layout()
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 16), 
                    textcoords='offset points', 
                    fontsize=24,
                    color="#FF914D",
                    fontname="Prata",
                    fontweight="bold",
                    alpha=1.0
                    )
    plt.show()


colonne_cat = ['CAEC', 'CALC', 'MTRANS']
dati_cat = X[colonne_cat].copy() 

plot_details_cat = {
    'CAEC': {
        'title_it': 'Consumo di Alimenti tra i Pasti', 
        'data_labels_en': ['no', 'Sometimes', 'Frequently', 'Always'],
        'tick_labels_it': ['Mai', 'A volte', 'Frequente', 'Sempre'] 
    },
    'CALC': {
        'title_it': 'Consumo di Alcol', 
        'data_labels_en': ['no', 'Sometimes', 'Frequently', 'Always'],
        'tick_labels_it': ['Mai', 'A volte', 'Frequente', 'Molto Frequente']
    },
    'MTRANS': {
        'title_it': 'Mezzo di Trasporto Principale', 
        'data_labels_en': ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'],
        'tick_labels_it': ['Automobile', 'Motocicletta', 'Bicicletta', 'Trasporto Pubblico', 'A Piedi']
    }
}

sns.set_style("white") 

for col, details in plot_details_cat.items():
    
    dati_cat[f'{col}_Codes'] = pd.Categorical(
        dati_cat[col], 
        categories=details['data_labels_en'], 
        ordered=True
    ).codes
    
    tick_positions = np.arange(len(details['tick_labels_it']))
    
    if col == "MTRANS" : 
        plt.figure(figsize=(9, 6), facecolor="#fff7eb")
        ax = sns.histplot(
        data=dati_cat,
        x=f'{col}_Codes',
        stat='count',
        discrete=True,
        color="#FFbc7e", 
        shrink=0.7,
        edgecolor= "#FFbc7e",
        alpha=1.0
        )
    elif col == "CALC" :
        plt.figure(figsize=(8, 5), facecolor="#fff7eb")
        ax = sns.histplot(
        data=dati_cat,
        x=f'{col}_Codes',
        stat='count',
        discrete=True,
        color="#b65337", 
        shrink=0.7,
        alpha=1.0,
        edgecolor= "#b65337"
        )
    else:
        plt.figure(figsize=(5, 5), facecolor="#fff7eb")
        ax = sns.histplot(
        data=dati_cat,
        x=f'{col}_Codes',
        stat='count',
        discrete=True,
        color="#b65337", 
        shrink=0.7,
        edgecolor= "#b65337",
        alpha=1.0
        )    
    
    plt.title(f"{details['title_it']}", fontsize=24, fontname='Prata', y = 1.1, color="#b65337",weight = "bold")
    plt.ylabel('')
    
    
    plt.xticks(
        tick_positions, 
        details['tick_labels_it'], 
        rotation=0, 
        ha='center', 
        fontsize=15,
        fontname="Prata", 
        color="#b65337"
    ) 
    plt.xlabel('')
    ax.set_facecolor("#fff7eb")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.get_yaxis().set_ticks([]) 
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 16), 
                        textcoords='offset points', 
                        fontsize=24,
                        fontname="Prata", 
                        color="#FF914D",
                        fontweight="bold",
                        )
    
    plt.tight_layout()
    plt.show()
