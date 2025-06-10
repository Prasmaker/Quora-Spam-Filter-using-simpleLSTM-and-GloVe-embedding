# Quora-Spam-Filter-using-simpleLSTM-and-GloVe-embedding
🎯 Project Objective
Built an intelligent spam filter to automatically identify spam questions on Quora using deep learning techniques with pre-trained word embeddings.
📊 Dataset & Preprocessing

Data Source: Quora questions dataset stored on Google Drive
Data Cleaning: Implemented text preprocessing including:

Removal of special characters, URLs, and extra whitespace
Text normalization and lowercasing
Handling missing values and empty questions


Data Splitting: Used 80-20 train-test split with train_test_split
Sequence Length: Optimized to max_len=44 based on question length analysis

🔧 Text Processing Pipeline

Tokenization: Used Keras Tokenizer with word-level tokenization
Vocabulary: Built vocabulary from training data using tk.fit_on_texts(x_train)
Sequence Conversion: Converted text to numerical sequences
Padding: Applied padding to ensure uniform sequence length of 44 tokens

🌐 Word Embeddings

GloVe Embeddings: Downloaded and integrated Stanford's GloVe 42B.300d embeddings

Source: http://nlp.stanford.edu/data/glove.42B.300d.zip
Dimensions: 300-dimensional word vectors
Coverage: 42 billion tokens for rich semantic representation


Embedding Integration: Created embedding matrix linking vocabulary to pre-trained vectors

🧠 Model Architecture
Built a deep learning model with:

Embedding Layer: Pre-trained GloVe embeddings (300 dimensions)
LSTM Layer: Long Short-Term Memory network for sequential pattern learning
Dense Layers: Fully connected layers for classification
Output: Sigmoid activation for binary classification (spam/not spam)
Optimization: Used Adam optimizer with appropriate learning rate

⚡ Training Optimization

Batch Size: Optimized to 2048 for faster training (64x speedup)
Class Imbalance: Handled using computed class weights with scikit-learn
Memory Management: Implemented strategies to prevent Colab session crashes
Early Stopping: Monitored training to prevent overfitting

📈 Model Performance
Achieved strong performance metrics:

ROC AUC: 0.9572 (Excellent discriminative ability)
Accuracy: 88.19% (Good overall performance)
Recall: 91.50% (Catches 91.5% of spam questions)
Precision: 33.55% (Conservative spam flagging)
F1-Score: 49.10% (Balanced measure)

🔍 Evaluation & Analysis

Comprehensive Metrics: Calculated accuracy, precision, recall, F1-score, ROC AUC
Confusion Matrix: Analyzed true/false positives and negatives
Threshold Analysis: Explored optimal decision thresholds
Model Interpretation: Understanding the precision-recall trade-off for spam detection

💾 Model Deployment Preparation

Model Saving: Saved trained model architecture and weights
Tokenizer Persistence: Saved tokenizer for future predictions
Backup Strategy: Implemented Google Drive backup for model artifacts
Prediction Pipeline: Created functions for real-time spam detection

🛠 Technical Stack

Deep Learning: TensorFlow/Keras
NLP: Pre-trained GloVe embeddings
Data Processing: Pandas, NumPy, Scikit-learn
Environment: Google Colab with GPU acceleration
Storage: Google Drive integration
