
# 🧠 duplicate question pair detection  
*author: rohit yadav*  

📓 **[open in google colab](https://colab.research.google.com/drive/1nYVucx_pr843eC9KNXRvO_dZ72ddMBU4?usp=sharing)**  

🌐 **try the live app (streamlit web demo)**  
👉 https://duplicate-question-pair-22gpnvme7lyw9gvnfqwa3z.streamlit.app/

this project focuses on detecting **duplicate question pairs**, similar to the **quora question pairs** problem.  
the goal is to identify whether two given questions have the same semantic meaning using **bag of words (bow)** features combined with advanced nlp-based preprocessing and feature engineering.

---

## 📘 project overview  

duplicate question detection is a common task in nlp where the system determines if two questions are semantically similar.  
this helps platforms like **quora**, **stack overflow**, and **reddit** reduce redundancy and improve user experience.

this project uses:
- **text preprocessing** (cleaning, stemming, stopword removal)
- **feature extraction** using **bag of words (bow)**
- **advanced features** like token overlap, fuzzy matching, and word length similarity
- a **machine learning model** trained on engineered features to predict whether two questions are duplicates

---

## 🧩 key features  

- ✅ cleaned and preprocessed text using **nltk / spacy**
- ✅ feature engineering for similarity scores (cosine, fuzzy ratio, etc.)
- ✅ bag-of-words representation
- ✅ model training and evaluation (accuracy, f1-score, precision, recall)
- ✅ pickled trained model for reuse (`model.pkl`)
- ✅ ready-to-run **google colab notebook** for full reproducibility

---

## 🛠️ technologies used  

| category           | tools & libraries           |
|-------------------|----------------------------|
| language           | python 3.x                |
| nlp                | nltk, spacy, fuzzywuzzy   |
| machine learning   | scikit-learn               |
| data handling      | pandas, numpy              |
| visualization      | matplotlib, seaborn        |
| others             | re, string, pickle         |

---

## 📂 project structure

```text
duplicate-question-pair-detection/
├── readme.md
├── requirements.txt
├── app.py
├── helper.py
├── model.pkl
├── cv.pkl
└── __pycache__/
```

---

## 🚀 how to run on google colab

1. **open the colab notebook**  
   👉 [click here to open](https://colab.research.google.com/drive/1nYVucx_pr843eC9KNXRvO_dZ72ddMBU4?usp=sharing)

2. **upload or mount your dataset** (if not already in the notebook)

3. **run all cells** to preprocess the data, extract features, and train the model

4. **load the trained model (optional)**  
   download and place the model file (`model.pkl`) in your colab environment if needed

---

## 📈 evaluation metrics

model performance is measured using:

- **accuracy**
- **precision**
- **recall**
- **f1-score**
- **confusion matrix visualization**

---

## 💡 future improvements

- integrate **tf-idf**, **word2vec**, or **sentence transformers** embeddings  
- build a **web interface** for real-time duplicate detection  
- use **deep learning models** like lstm or bert for improved semantic understanding  

---

## 🙌 acknowledgements

- dataset: quora question pair  
- libraries: scikit-learn, nltk  


 
