# Analisi e classificazione dei livelli di obesità tramite dati comportamentali e fisici
Principi e modelli della Percezione - A.A. 2025/2026

A cura di: Daniele Preda, Luca Balconi, Riccardo Romagnolo

## Introduzione
Questo progetto è dedicato all'analisi, alla visualizzazione e alla modellazione predittiva basata sul dataset **"Estimation of Obesity Levels based on Eating Habits and Physical Condition"** fornito dall'UCI Machine Learning Repository.

L'obiettivo principale è:
1. **Esplorare e visualizzare** le relazioni tra le abitudini alimentari, le condizioni fisiche e il livello di obesità.
2. Sviluppare un modello di **Random Forest** per la **classificazione** del livello di obesità di un individuo.

## Il dataset
Il dataset utilizzato è presente a questo [link](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
* **Caratteristiche:** Il set di dati include variabili anagrafiche (età, sesso, altezza, peso) e diverse abitudini comportamentali e stili di vita (attività fisica, consumo di acqua, consumo di alcol).
* **Target (Variabile da prevedere):** `NObesity` (Livello di obesità), che mostra se l'individuo è sottopeso/normopeso oppure sovrappeso/obeso.

## Progetto
1. **Visualizzazione:** all'interno della directory "visualizzazione" sono presenti i file in python, divisi per dominio delle feature (categoriche, binarie o target) e la heatmap.
2. **Classificazione:** nella directory "random_forest" è presente un file python che addestra l'algoritmo e mostra la sua applicazione sul dataset, mostrando:
   * **Feature importance** delle abitudini che influiscono di più sull'essere categorizzato come persona sovrappesa o obesa
   * L'**albero decisionale** generato sulla base delle feature più impattanti
3. **Valutazione del modello:** nel file randomForest.py sono anche presenti le valutazioni del modello scaturite dall'addestramento e dall'utilizzo dell'algoritmo.
