# Predicting missing annotations in Gene Ontology with Knowledge Graph Embeddings and True Path Rule

 
 We propose a method for training Knowledge Graph Embeddings (KGEs) that incorporates True Path Rule (TPR) to make the learned embeddings more suitable for inference rules.

 src folder contains all necessary py files for implementing TransE+TPR and generating datasets.

 * utils.py: All necessary functions for dataset creation in dataset_generation.ipynb
 * loaders.py, model_utils.py, models.py: All necessary functions for building, training, testing and evaluating TransE and TransE+TPR methods

 experiments folder contains the real reasoner runs behind the paper's Scenario 1 results,
 one notebook per reported dataset pair (GOA18-19 through GOA21-22). It also contains one
 exploratory notebook (train&test_generation_sc2-2_TransE&GOA17-18.ipynb) for an
 unreported dataset pair (GOA17-18), showing Scenario 2.2 test/valid-set construction only.

 The paper: [Predicting missing annotations in Gene Ontology with Knowledge Graph Embeddings and True Path Rule](https://ceur-ws.org/Vol-3415/paper-9.pdf)

