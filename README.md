# Sentence Contradiction Classification

## Project Overview

This project focuses on classifying pairs of sentences into three categories: contradiction, entailment, or neutral. The goal is to determine the relationship between two sentences (a premise and a hypothesis) and predict whether they contradict each other, entail each other, or are neutral. This task is a common problem in natural language processing (NLP) and is often used to evaluate the understanding of language models.

The project involves data preprocessing, model implementation, training, and evaluation. Various machine learning and deep learning models are explored, including traditional classifiers like Random Forest and Decision Trees, as well as advanced transformer-based models like BERT and XLM-Roberta.

## Dataset Description

The dataset used in this project is a collection of sentence pairs labeled with their relationship. Each pair consists of a premise and a hypothesis, and the label indicates whether the hypothesis contradicts, entails, or is neutral with respect to the premise. The dataset contains 12,120 entries and includes sentences in multiple languages, making it a multilingual dataset.

### Dataset Columns:
- **id**: Unique identifier for each sentence pair.
- **premise**: The first sentence (premise).
- **hypothesis**: The second sentence (hypothesis).
- **lang_abv**: Language abbreviation (e.g., 'en' for English).
- **language**: Full language name (e.g., 'English').
- **label**: The relationship label (0: contradiction, 1: entailment, 2: neutral).

## Model Implementation Details

The project explores several models for sentence contradiction classification:

1. **Traditional Machine Learning Models**:
   - **Random Forest Classifier**
   - **Decision Tree Classifier**
   - **XGBoost Classifier**

2. **Deep Learning Models**:
   - **LSTM (Long Short-Term Memory)**
   - **GRU (Gated Recurrent Unit)**
   - **BERT (Bidirectional Encoder Representations from Transformers)**
   - **XLM-Roberta (Cross-lingual Language Model)**

### Preprocessing:
- Tokenization and cleaning of text data.
- Handling of multilingual text using appropriate tokenizers.
- Splitting the dataset into training and testing sets.

### Model Training:
- Traditional models are trained using TF-IDF vectorized features.
- Deep learning models are trained using pre-trained embeddings (e.g., Word2Vec, BERT embeddings).
- Hyperparameter tuning is performed using RandomizedSearchCV for traditional models.

### Evaluation Metrics:
- **Accuracy**
- **Classification Report (Precision, Recall, F1-Score)**
- **Confusion Matrix**
- **ROC Curve and AUC**

## Steps to Run the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sentence-contradiction-classification.git
   cd sentence-contradiction-classification
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Place the dataset (`train.csv`) in the project directory.

4. **Run the Jupyter Notebook**:
   Open the provided Jupyter Notebook (`Sentence Contradiction Classification.ipynb`) and run the cells sequentially to preprocess the data, train the models, and evaluate their performance.

5. **Model Training and Evaluation**:
   - The notebook includes code for training and evaluating both traditional and deep learning models.
   - You can modify the hyperparameters or experiment with different models as needed.

6. **View Results**:
   The notebook will output the evaluation metrics for each model, including accuracy, classification reports, and confusion matrices.

## Model Evaluation Results

The evaluation results for the models are as follows:

### Traditional Models:
- **Random Forest Classifier**: Achieved an accuracy of 34%.
- **Decision Tree Classifier**: Achieved an accuracy of 35%.
- **XGBoost Classifier**: Achieved an accuracy of 35%.

### Deep Learning Models:
- **LSTM**: Achieved an accuracy of 35%.

### Observations:
- Transformer-based models (BERT, XLM-Roberta) generally outperformed traditional models due to their ability to capture contextual information.
- The multilingual nature of the dataset posed challenges, but XLM-Roberta handled it effectively due to its cross-lingual capabilities.

## Additional Observations and Notes

- **Multilingual Challenges**: The dataset includes sentences in multiple languages, which adds complexity to the task. Models like XLM-Roberta, which are designed for cross-lingual tasks, performed better on multilingual data.
- **Hyperparameter Tuning**: RandomizedSearchCV was used to optimize hyperparameters for traditional models, leading to improved performance.
- **Data Imbalance**: The dataset may have class imbalance issues, which could affect model performance. Techniques like oversampling or class weighting could be explored to address this.
- **Future Work**: Further improvements could be made by fine-tuning transformer models on specific languages or using ensemble methods to combine the strengths of different models.

## Conclusion

This project demonstrates the effectiveness of various machine learning and deep learning models for sentence contradiction classification. The results highlight the superiority of transformer-based models, especially in handling multilingual datasets. Future work could focus on improving model performance through advanced techniques and addressing data imbalance issues.
