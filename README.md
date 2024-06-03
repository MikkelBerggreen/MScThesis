# Domain Adaptation of Convolutional Neural Networks in Neuroimaging through EEG Signal Reconstruction
Thesis in Master of Computer Science at the IT University of Copenhagen, Spring 2024. Authored by Mikkel Rahlff Bergreen and Simon Brage Petersen. Supervised by Paolo Burelli and Laurits Dixen of the [ITU brAIn lab](https://brainlab.itu.dk/).

## Abstract

This thesis explores the intersection of neuroimaging and deep learning, through a domain adaptation of convolutional neural networks (CNNs) that investigates their effectiveness in reconstructing EEG signals from visual stimuli. We employ three different model designs for the image-to-EEG task: MLPEEG, a simple MLP acting as a baseline; ResNetEEG, the ResNet-18 architecture modified to predict EEG directly; and ResNetLatentEEG, a novel approach of combining the image feature extraction of ResNet-18 with the decoding capabilities of a temporal convolutional network autoencoder. We evaluate their performance based on the encoding fidelity, with three metrics capturing separate relations, and the encoding similarity with representational similarity analysis. The results highlight significant challenges in EEG reconstruction from visual stimuli using our approach, with marginal encoding fidelity and virtually zero correlation in encoding similarity across all models.
Due to the poor encoding performance, we cannot conclude exactly how our different model configurations impact the task of reconstructing EEG signals. However, we do not see any indications that the feature extraction of CNNs specifically poses a limitation in this context. The models display the ability to learn from the training data but are limited by overfitting and lack of generalization. For future work, we stress the importance of further research into metrics that can provide meaningful and interpretable comparisons of EEG signals, and suggest the need for a more systematic approach to model architecture experimentation.

# Repository structure

**NOTE!** This repository only contains the source code used in this project, and as such, the structure of this repository does not reflect the sturcture shown below. The full repository containing all data, source code and trained models is available through the repository on the ITU Enterprise Github only: [https://github.itu.dk/sibp/Thesis](https://github.itu.dk/sibp/Thesis).

    .
    ├── 📁 data                    # Raw EEG files and image metadata
    ├── 📁 preprocessed stimuli    # Preprocessed image files
    ├── 📁 src                     # Source files
    │   ├── 📁 core                  # Logic for training, testing and handling data
    │   ├── 📁 metrics               # Loss functions and performance related metrics (Algonaut Challenge Score, RSA, etc.)
    │   ├── 📁 models                # Logic for setting up models
    │   ├── 📁 utils                 # Various utility classes and functions (config handling, visualizations, etc.)
    │   └── 📄 main.ipynb            # Notebook containing code for most of the visualizations and evaluations from the report
    ├── 📁 stimuli                 # Raw image files
    └── 📁 trained_models          # Trained models (model naming convention described below)
        ├── 📁 3_channels            # Models trained on the 3-channel subset: { O1, Oz, O2 }
        ├── 📁 5_channels            # Models trained on the 5-channel subset: { O1, Oz, O2, T7, T8 }
        └── 📁 1_channels            # Models trained on the 11-channel subset: { O1, Oz, O2, T7, T8, TP7, TP8, P7, P8, PO7, PO8 }
        
# Model naming conventions

The models contained in the `trained_models`-folders are named after the following convention:

- 1: Model name (Note: `MLPEEG` is listed as `BaselineEEGReconstruction`, `ResNetEEG` as `ResNet`, and `ResNetLatentEEG` as it is)
- 2: Shorthands for the loss function used during training (`mse` for MSELoss and `wd` for approximate Wasserstein (Sinkhorn) distance)
- 3: Number of ResNet modules used during training, ranging 0-4 (if applicable)
- 4: Date and time of the run

Full formatting of the model naming convention:

`{1}_{2}{3}_{4}.pth`

Examples:
`ResNet_mse4_13_05_2024__14_16.pth`,
`ResNet_wd2_15_05_2024__18_01.pth`
