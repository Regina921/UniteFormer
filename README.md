# UniteFormer: Unifying Node and Edge Modalities in Transformers for Vehicle Routing Problems 

The PyTorch Implementation of NeurIPS 2025 --"[UniteFormer: Unifying Node and Edge Modalities in Transformers for Vehicle Routing Problems](https://openreview.net/pdf?id=BRklmFlCsD)."

 Neural solvers for the Vehicle Routing Problem (VRP) have typically relied on either node or edge inputs, limiting their flexibility and generalization in real-world
 scenarios. We propose UniteFormer, a unified neural solver that supports node only, edge-only, and hybrid input types through a single model trained via joint edge-node modalities. UniteFormer introduces: (1) a mixed encoder that integrates graph convolutional networks and attention mechanisms to collaboratively process node and edge features, capturing cross-modal interactions between them; and (2) a parallel decoder enhanced with query mapping and a feed-forward layer for improved representation. The model is trained with REINFORCE by randomly sampling input types across batches. These results underscore UniteFormer’s ability to handle diverse input modalities and its strong potential to improve performance across various VRP tasks.

## Overview

<img width="1523" height="816" alt="image" src="https://github.com/user-attachments/assets/45bc120d-16b0-4444-b04a-db29e9ec46ac" />



## Dependencies
```bash
Python >= 3.8
Pytorch >= 2.0.1
numpy==1.24.4
matplotlib==3.5.2 
tqdm==4.67.1
```

## Download datasets and models

Download `datasets` and `models` from [Hugging Face](https://huggingface.co/datasets/Regina921/UniteFormer/tree/main). 

Unzip `UF-TSP-results.zip` and `UF-CVRP-results.zip`, and organize the files in the project directory as follows:

```bash

UniteFormer
├─ UF-TSP
│  ├─ data
│  └─ train_models
└─ UF-CVRP
   ├─ data
   └─ train_models

```


## Citation


 
## Acknowledgments

* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/yd-kwon/POMO
* https://github.com/yd-kwon/MatNet 
 
