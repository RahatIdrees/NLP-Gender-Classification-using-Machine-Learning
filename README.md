<h1 align="center"> Blog Author Gender Classification using NLP</h1>

<p align="center">
Machine Learning & Natural Language Processing Project
</p>

---

##  Introduction

This project explores how Natural Language Processing (NLP) and Machine Learning can be used to identify the gender of blog authors based on their writing patterns and textual style.

Different text representation techniques and classification models were tested to determine the most accurate approach for gender prediction.

---

##  Project Goal

The main objective of this project is to build an intelligent text classification system capable of predicting whether a blog post was written by a male or female author.

The solution can support:
- Recommendation systems
- Social media analytics
- User behavior analysis
- Personalized marketing
- Human-computer interaction systems

---

##  Tools & Libraries

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Scikit-learn | Machine Learning |
| Pandas & NumPy | Data Processing |
| Matplotlib | Visualization |
| Google Colab | Development Environment |

---

##  Dataset Details

The dataset consists of two main attributes:

- **BLOG** → Text written by blog authors
- **GENDER** → Author gender label

This is a supervised binary classification task.

---

##  Text Preprocessing

To improve model performance, several preprocessing techniques were applied:

✔️ Lowercase conversion  
✔️ Removal of punctuation  
✔️ Removal of numbers  
✔️ Removal of special characters  
✔️ Missing value handling  
✔️ Label encoding  

### Encoded Labels
| Gender | Value |
|---|---|
| Female | 0 |
| Male | 1 |

---

##  Dataset Splitting

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

This split helps evaluate model generalization on unseen data.

---

##  Feature Extraction Techniques

### 🔹 Count Vectorization
Transforms text into numerical vectors using word occurrence frequency.

### 🔹 TF-IDF Representation
Assigns importance scores to words based on document frequency.

### 🔹 TF-IDF + PCA
Reduces feature dimensionality while preserving important information.

---

##  Classification Models

Three machine learning algorithms were trained and evaluated:

### Logistic Regression
A simple and efficient baseline algorithm commonly used for text classification.

### Support Vector Machine (SVM)
A powerful classifier that performs well on high-dimensional NLP datasets.

### Random Forest
An ensemble-based algorithm that improves stability and reduces overfitting.

---

##  Model Evaluation

Performance was measured using:

- Accuracy
- Precision
- Recall
- F1-Score

---

##  Experimental Results

| Model | Accuracy | F1-Score |
|---|---|---|
| TF-IDF + Logistic Regression | 67.30% | 0.679 |
| TF-IDF + SVM | **67.69%** | **0.681** |
| TF-IDF + Random Forest | 64.80% | 0.636 |

📌 **Best Performing Model:** TF-IDF with Support Vector Machine (SVM)


---

##  Final Outcome

The experimental analysis shows that NLP-based feature extraction combined with machine learning can effectively classify blog author gender.

Among all tested approaches, the TF-IDF + SVM model achieved the strongest performance and produced the highest classification accuracy.


