# Étiqueteur POS avec réseaux récurrents
Dans ce projet, j’utilise un réseau de neurones récurrent de type GRU, implémenté avec PyTorch, pour réaliser l’étiquetage morpho-syntaxique (POS tagging). Les données utilisées proviennent du corpus Sequoia au format CoNLL-U

Ce Tp contient :
- un notebook d'entraînement `train_postag.ipynb` pour l'enrainement du modèl 
- un script de prédiction `predict_postag.py`pour son évaluation


## Prérequis
- le script d'évaluation `lib/evaluate.py`
- un script `lib/conllulib`
- PyTorch
- La librairie `conllu`


## Données
Les fichiers Sequoia en format CoNLL-U doivent être disponibles dans le dossier sequoia  :
- `sequoia/sequoia-ud.parseme.frsemcor.simple.train`
- `sequoia/sequoia-ud.parseme.frsemcor.simple.dev`

## Entraîner le modèle
Ouvrir et exécuter le notebook `train_postag.ipynb`.

À la fin de l'entraînement, le modèle est sauvegardé ici :
- `resultats/postagger.pt`

et l'historique ici :
- `resultats/hist_postagger.json` ça pour garder les résultats pour faire des comparaison ou des analyses éventuellement. 

## Faire des prédictions
Lancer le script de prédiction
```bash
python predict_postag.py
```

Le script charge `resultats/postagger.pt` et produit un fichier de sortie  :
- `resultats/sequoia-ud.parseme.frsemcor.simple-rnn-dev.pred`

## Évaluer les prédictions
Commande d'évaluation :

```bash
python lib/evaluate.py -p resultats/sequoia-ud.parseme.frsemcor.simple-rnn-dev.pred -g sequoia/sequoia-ud.parseme.frsemcor.simple.dev -t sequoia/sequoia-ud.parseme.frsemcor.simple.train -c upos -f form
```

