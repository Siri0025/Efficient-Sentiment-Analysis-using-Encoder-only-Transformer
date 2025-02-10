# Efficient-Sentiment-Analysis-using-Encoder-only-Transformer
This repository contains an efficient sentiment analysis model built using an Encoder-only Transformer architecture. The implementation leverages Optuna for hyperparameter optimization, ensuring optimal model performance. 

## **Overview**
This project implements an **efficient sentiment analysis model** using an **Encoder-only Transformer** architecture. The goal is to classify sentiment from text data while optimizing model performance using **Optuna for hyperparameter tuning**. The implementation includes **dataset preprocessing, model training, evaluation, and hyperparameter optimization**, ensuring a structured and scalable approach for sentiment classification.

## **Features**
- **Encoder-only Transformer** for sentiment analysis.
- **Data Preprocessing** and efficient dataset handling.
- **Training and Evaluation** pipeline.
- **Hyperparameter Optimization** using Optuna.
- **Modular Code Structure** for easy scalability and experimentation.

## **Project Structure**
```
├── Transformer_build.py   # Implements the Encoder-only Transformer model
├── Train_build.py         # Handles data loading, training, and evaluation
├── Main.py                # Implements Optuna for hyperparameter tuning
├── requirements.txt       # Lists dependencies for easy installation
├── README.md              # Project documentation
└── data/                  # Directory containing dataset (if applicable)
```

### **1. Transformer_build.py**
This file contains the **Encoder-only Transformer** architecture, which is responsible for processing input text and extracting meaningful features for sentiment classification. It includes:
- Token embedding and positional encoding.
- Multi-head self-attention and feedforward layers.
- Layer normalization and dropout for regularization.

### **2. Train_build.py**
This file handles **data loading, training, and evaluation**. It includes:
- **Dataset functions** for text preprocessing (tokenization, padding, and batching).
- **Training loop** for optimizing the model.
- **Evaluation functions** to measure accuracy and other performance metrics.

### **3. Main.py**
This file integrates **Optuna** to perform **automated hyperparameter tuning**, optimizing parameters like:
- Learning rate.
- Batch size.
- Number of layers and attention heads.
- Dropout rate.

## **Installation**
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/Efficient-Sentiment-Transformer.git
cd Efficient-Sentiment-Transformer
pip install -r requirements.txt
```

## **Usage**
### **1. Train the Model**
To train the sentiment analysis model, run:
```bash
python Train_build.py
```

### **2. Run Hyperparameter Optimization**
To find the best hyperparameters using Optuna, execute:
```bash
python Main.py
```

## **Results & Evaluation**
- The model is evaluated on standard metrics such as **accuracy, precision, recall, and F1-score**.
- Optuna helps to automatically find the best hyperparameters for improved performance.
- The **trained model** can be used for further sentiment classification tasks.

## **Future Enhancements**
- Implementing **pre-trained transformer models** (e.g., BERT, RoBERTa) for better accuracy.
- Extending the project to support **multi-class sentiment analysis**.
- Exploring **different attention mechanisms** to improve model interpretability.

## **Contributing**
Contributions are welcome! Feel free to **fork the repository**, create a **feature branch**, and submit a **pull request**.

