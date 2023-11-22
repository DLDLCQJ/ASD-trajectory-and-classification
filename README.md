## Introduction to Distinct Neurodevelopmental Patterns and Intermediate Fusion-Based Predictive Modeling in Autism

This is the origin Pytorch implementation of LMFGCN in the following paper:
"Distinct Neurodevelopmental Patterns and Intermediate Fusion-Based Predictive Modeling in Autism"

The project focuses on utilizing advanced machine learning techniques to analyze neurodevelopmental patterns associated with Autism Spectrum Disorder (ASD). Our approach leverages a multi-view learning framework to integrate diverse neuroimaging and clinical data, aiming to enhance the understanding and predictive modeling of ASD.

## Key Features
- Multi-View Learning Framework: Implements a sophisticated machine learning model to handle multiple data types simultaneously, providing a holistic view of the underlying neurodevelopmental patterns in ASD.
- Graph Convolutional Networks (GCN): Employs GCN for efficient processing and integration of neuroimaging data, capturing the complex spatial relationships inherent in brain imaging.
- Low-rank Multimodal Fusion (LMF): Utilizes LMF techniques for integrating multimodal data sources, ensuring a comprehensive analysis that accounts for different aspects of ASD.
- Early Stopping Mechanism: Incorporates an early stopping strategy to prevent overfitting, ensuring robust and generalizable model performance.
- Reproducibility and Rigorous Evaluation: The codebase includes functions for reproducibility and a robust evaluation framework using Stratified K-Fold cross-validation.

## Repository Structure
- models/: Contains the implementation of GCN, ChebConv, and other neural network architectures.
- utils/: Utility functions for data preprocessing, performance metric computation, and model evaluation.
- data/: Scripts and instructions for preprocessing and handling the dataset used in the study.
- training_scripts/: Scripts to run the training, validation, and testing phases of the model.
- results/: Folder to store the output results, including model performance metrics and plots.

## Citation
If you find this work useful in your research, please consider citing the original paper:
"Distinct Neurodevelopmental Patterns and Intermediate Fusion-Based Predictive Modeling in Autism"

## Contact
For any queries or discussions regarding this project, feel free to open an issue in this repository or contact the maintainers directly.
