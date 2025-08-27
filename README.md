# Machine Learning Tasks - Supervised and Unsupervised Learning

This repository contains implementations of various machine learning tasks covering both supervised and unsupervised learning approaches using Python and scikit-learn.

## 📋 Overview

This project demonstrates practical applications of machine learning algorithms including:
- **Supervised Learning**: Classification and Regression tasks
- **Unsupervised Learning**: Clustering and pattern recognition
- **Natural Language Processing**: Text classification and clustering
- **Predictive Modeling**: Time prediction and product analysis

## 🚀 Tasks Implemented

### Task 1: Message Intent Classifier (Supervised Learning)
**Objective**: Train a model to classify customer service message intents

**Features**:
- Text preprocessing with regex
- CountVectorizer for text vectorization
- Multinomial Naive Bayes classifier
- Interactive testing interface
- Expanded dataset with 17 training examples

**Categories**: pedido, suporte, promoção, informação, pagamento, cancelamento, rastreamento, troca, devolução, conta, entrega

### Task 2: Academic Support Bot (Supervised Learning)
**Objective**: Create a classifier for academic institution messages

**Features**:
- Academic-specific dataset with 15 training examples
- Categorization of student queries
- Real-time message classification

**Categories**: matricula, avaliacao, biblioteca, documentos, sistema, tcc, transferencia, eventos, estagio, secretaria, bolsa, formatura

### Task 3: Pizza Delivery Time Prediction (Supervised Learning)
**Objective**: Predict delivery time based on distance and order size

**Features**:
- Linear Regression model
- Multi-feature prediction (distance + pizza count)
- Expanded training dataset with 10 examples
- Real-time prediction testing

**Input Features**: Distance (km), Number of pizzas
**Output**: Delivery time (minutes)

### Task 4: Message Clustering (Unsupervised Learning)
**Objective**: Group similar messages without predefined categories

**Features**:
- K-Means clustering with 3 clusters
- Text vectorization for similarity analysis
- Interactive message classification
- Expanded dataset with 18 messages

### Task 5: Tourism Chatbot Clustering (Unsupervised Learning)
**Objective**: Group tourism-related queries into meaningful categories

**Features**:
- K-Means clustering with 4 clusters
- Tourism-specific dataset with 20 queries
- Automatic categorization of travel services

**Categories**: Accommodation, Transportation, Food & Dining, Tours & Activities

### Task 6: Product Anchor Analysis (Unsupervised Learning)
**Objective**: Identify optimal "anchor" products for homepage featuring

**Features**:
- K-Means clustering for product segmentation
- Multi-dimensional analysis (price + popularity)
- Strategic product positioning insights

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computations
- **Regular Expressions**: Text preprocessing

### Key Libraries
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
import re
```

## 📁 Project Structure

```
├── README.md                           # This file
├── Missoes.txt                         # Original task specifications
└── 27.08.2025 Respostas 1-6.py        # Complete implementation
```

## 🎯 Key Learning Outcomes

### Supervised Learning
- **Text Classification**: Understanding how to process and classify text data
- **Regression Analysis**: Predicting continuous values from multiple features
- **Model Training**: Proper dataset preparation and model fitting
- **Feature Engineering**: Text vectorization and preprocessing

### Unsupervised Learning
- **Clustering**: Grouping similar data points without labels
- **Pattern Recognition**: Discovering hidden structures in data
- **Dimensionality Reduction**: Working with high-dimensional text data
- **Data Exploration**: Understanding data distribution and relationships

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**:
   ```bash
   pip install scikit-learn numpy
   ```

3. **Run the complete implementation**:
   ```bash
   python "27.08.2025 Respostas 1-6.py"
   ```

## 📊 Example Outputs

### Task 1 - Message Classification
```
'Quero comprar algo' -> Intenção: pedido
'Preciso de ajuda urgente' -> Intenção: suporte
'Tem cupom de desconto?' -> Intenção: promoção
```

### Task 3 - Delivery Prediction
```
Tempo de entrega previsto para o novo pedido: 35.00 minutos
Pedido 1: 6 km, 3 pizza(s) -> 37.50 minutos
```

### Task 4 - Message Clustering
```
'Quero pedir pizza' => Cluster 0
'Preciso de suporte no aplicativo' => Cluster 1
'Vocês têm sobremesas?' => Cluster 2
```

## 🔍 Technical Details

### Text Preprocessing
- Convert to lowercase
- Remove punctuation and numbers
- Strip extra whitespace
- Regular expression cleaning

### Model Parameters
- **K-Means**: n_clusters=3/4, random_state=42, n_init=10
- **Naive Bayes**: MultinomialNB with default parameters
- **Linear Regression**: Standard implementation

### Data Validation
- All models include test cases
- Interactive interfaces for real-time testing
- Comprehensive error handling

## 📈 Performance Insights

- **Classification Accuracy**: Models show good performance on test data
- **Clustering Quality**: Clear separation of message categories
- **Prediction Reliability**: Linear regression provides reasonable delivery estimates
- **Scalability**: Code structure allows easy dataset expansion

## 🤝 Contributing

This project was developed as part of a machine learning course. Feel free to:
- Improve the algorithms
- Add more training data
- Enhance the preprocessing steps
- Implement additional evaluation metrics

## 📝 License

This project is for educational purposes. All code is open source and available for learning and modification.

## 👨‍💻 Author

Developed as part of machine learning coursework focusing on practical applications of supervised and unsupervised learning techniques.

---

**Note**: This repository contains complete implementations of all 6 machine learning tasks as specified in the original requirements. All code has been tested and validated for proper execution.
