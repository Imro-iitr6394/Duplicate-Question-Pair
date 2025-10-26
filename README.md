# 🧠 Duplicate Question Pair Detection  
*Author: Rohit Yadav*  

📓 **[Open in Google Colab](https://colab.research.google.com/drive/1nYVucx_pr843eC9KNXRvO_dZ72ddMBU4?usp=sharing)**  

This project focuses on detecting **duplicate question pairs**, similar to the **Quora Question Pairs** problem.  
The goal is to identify whether two given questions have the same semantic meaning using **Bag of Words (BoW)** features combined with advanced NLP-based preprocessing and feature engineering.

---

## 📘 Project Overview  

Duplicate question detection is a common task in NLP where the system determines if two questions are semantically similar.  
This helps platforms like **Quora**, **Stack Overflow**, and **Reddit** reduce redundancy and improve user experience.

This project uses:
- **Text preprocessing** (cleaning, stemming, stopword removal)
- **Feature extraction** using **Bag of Words (BoW)**
- **Advanced features** like token overlap, fuzzy matching, and word length similarity
- A **Machine Learning model** trained on engineered features to predict whether two questions are duplicates.

---

## 🧩 Key Features  

- ✅ Cleaned and preprocessed text using **NLTK / SpaCy**
- ✅ Feature engineering for similarity scores (cosine, fuzzy ratio, etc.)
- ✅ Bag-of-Words representation
- ✅ Model training and evaluation (accuracy, F1-score, precision, recall)
- ✅ Pickled trained model for reuse (`model.pkl`)
- ✅ Ready-to-run **Google Colab Notebook** for full reproducibility

---

## 🛠️ Technologies Used  

| Category | Tools & Libraries |
|-----------|------------------|
| Language | Python 3.x |
| NLP | NLTK, SpaCy, FuzzyWuzzy |
| Machine Learning | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Others | re, string, pickle |

---



---

## 📂 Project Structure

```text
Duplicate-Question-Pair-Detection/
├── README.md
├── requirements.txt
├── app.py
├── helper.py
├── model.pkl
├── cv.pkl
└── __pycache__/







---

## 🚀 How to Run on Google Colab  
1. **Open the Colab notebook**  
   👉 [Click here to open](https://colab.research.google.com/drive/1nYVucx_pr843eC9KNXRvO_dZ72ddMBU4?usp=sharing)

2. **Upload or mount your dataset** (if not already in the notebook)

3. **Run all cells** to preprocess the data, extract features, and train the model.

4. **Load the trained model (optional)**  
   Download and place the model file in your Colab environment:

   
---

## 📈 Evaluation Metrics  

The model is evaluated on multiple metrics for better interpretability:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix Visualization**

---

## 💡 Future Improvements  

- Integrate **TF-IDF**, **Word2Vec**, or **Sentence Transformers** embeddings  
- Build a **web interface** for real-time duplicate detection  
- Use **deep learning models** like LSTM or BERT for improved semantic understanding  

---

## 🙌 Acknowledgements  

- [Quora Question Pair Dataset](https://www.kaggle.com/c/quora-question-pairs)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [NLTK Documentation](https://www.nltk.org/)  

---



 
