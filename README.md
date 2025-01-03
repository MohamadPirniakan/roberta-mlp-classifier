# Medical Specialty Classification using NLP and MLP

This repository contains an end-to-end pipeline for classifying medical transcriptions into various medical specialties. The solution leverages RoBERTa embeddings for feature extraction and a Multi-Layer Perceptron (MLP) for classification. It also addresses class imbalance using SMOTE and provides a complete workflow for training, evaluation, and visualization.

---

## Features
- **Text Embeddings**: RoBERTa is used to generate embeddings from medical transcriptions.
- **Class Imbalance Handling**: SMOTE is applied to balance the dataset.
- **Custom Classifier**: A Multi-Layer Perceptron (MLP) is implemented for classification.
- **Visualization**: Confusion matrix and class distribution are visualized.
- **GPU Support**: The model leverages GPU acceleration when available.

---

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Pipeline](#pipeline)
5. [Results](#results)
6. [License](#license)

---

## Installation
To run this project, ensure you have Python 3.8 or higher installed. Then, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/MohamadPirniakan/roberta-mlp-classifier.git
cd medical-specialty-classification

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset
The dataset used in this project is `mtsamples.csv`, which contains medical transcriptions labeled with their corresponding medical specialties. Ensure the dataset is placed in the root directory.

---

## Usage

### Data Preprocessing and Embedding Generation
Run the script to preprocess the data, generate embeddings, and handle class imbalance:

```bash
python preprocess.py
```

### Training the Model
Train the MLP model using the generated embeddings:

```bash
python train.py
```

### Evaluation
Evaluate the trained model on the test set:

```bash
python evaluate.py
```

---

## Pipeline

1. **Data Preprocessing**
   - Handle missing values.
   - Encode medical specialties into numeric labels.

2. **Embedding Generation**
   - Tokenize transcriptions using RoBERTa.
   - Generate text embeddings.

3. **Class Balancing**
   - Apply SMOTE to address class imbalance.

4. **Training**
   - Train an MLP classifier with a custom architecture.

5. **Evaluation**
   - Generate classification metrics and confusion matrix.

---

## Results

### Classification Metrics
Example classification report:

```
              precision    recall  f1-score   support

   Specialty1       0.89      0.90      0.89       150
   Specialty2       0.85      0.84      0.85       120
   ...

   accuracy                           0.87       500
  macro avg       0.87      0.87      0.87       500
 weighted avg    0.87      0.87      0.87       500
```

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Hugging Face for the RoBERTa model.
- scikit-learn and imbalanced-learn for data preprocessing and SMOTE.
- PyTorch for model implementation.

---

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.
