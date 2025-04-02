# Classificazione MRI Alzheimer con PyTorch

## Descrizione
Questo progetto implementa un modello di rete neurale per classificare immagini di risonanza magnetica (MRI) in diverse fasi della malattia di Alzheimer. Il modello è in grado di distinguere tra pazienti non dementi, con demenza molto lieve, lieve o moderata, utilizzando una rete neurale addestrata con PyTorch.

## Tecnologie Utilizzate
- **Python**: Elaborazione delle immagini, addestramento del modello e valutazione
- **Librerie Python**: PyTorch, NumPy, Pandas, Scikit-image, Matplotlib

## Risultati
Il modello raggiunge un'elevata accuratezza nella classificazione delle immagini MRI nelle diverse categorie di demenza. I risultati sono visualizzati attraverso grafici di perdita e accuratezza durante l'addestramento, insieme a una matrice di confusione e un rapporto di classificazione dettagliato.

## Come Iniziare
Le istruzioni per accedere al dataset, eseguire l'analisi ed esplorare le dashboard interattive sono disponibili nei file del progetto.

## Struttura del Progetto
- `PyTorch_Model.py`: Script Python con il codice del modello
- `results/`: Cartella contenente i risultati dell'analisi
  - `MRI_random_sample.png`: Visualizzazione di campioni casuali di immagini MRI
  - `training_metrics.txt`: Log delle metriche per epoca durante l'addestramento
  - `training_curves.png`: Grafici dell'andamento di perdita e accuratezza durante l'addestramento
  - `testing_metrics.txt`:Risultati finali del modello sul set di test (96.48% accuratezza)
  - `confusion_matrix.png`: Matrice di confusione delle previsioni del modello

## Funzionalità
- Preprocessamento delle immagini MRI
- Visualizzazione di esempi di immagini con le relative etichette
- Addestramento di un modello di rete neurale
- Monitoraggio delle prestazioni durante l'addestramento
- Valutazione dettagliata del modello finale


<br><br><br><br><br>


# Alzheimer MRI Classification with PyTorch

## Description
This project implements a neural network model to classify MRI images into different stages of Alzheimer's disease. The model is capable of distinguishing between non-demented patients and those with very mild, mild ore moderate dementia, using a neural network trained with PyTorch.

## Technologies Used
- **Python**: Image processing, model training, and evaluation
- **Python Libraries**: PyTorch, NumPy, Pandas, Scikit-image, Matplotlib

## Results
The model achieves high accuracy in classifying MRI images into different dementia categories. Results are visualized through loss and accuracy plots during training, along with a confusion matrix and detailed classification report.

## Getting Started
Instructions for accessing the dataset, running the analysis, and exploring the interactive dashboards are available in the project files.

## Project Structure
- `PyTorch_Model.py`: Python script with model code
- `results/`: Folder containing analysis results
  - `MRI_random_sample.png`: Visualization of random MRI image samples
  - `training_metrics.txt`: Epoch-by-epoch metrics log during training
  - `training_curves.png`: Plots of loss and accuracy during training
  - `testing_metrics.txt`: Final model results on the test set (96.48% accuracy)
  - `confusion_matrix.png`: Confusion matrix of model predictions

## Features
- MRI image preprocessing
- Visualization of sample images with their labels
- Training of a neural network model
- Performance monitoring during training
- Detailed evaluation of the final model
