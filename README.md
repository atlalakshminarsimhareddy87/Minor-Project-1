❤️ Heart Risk Prediction

This project is designed to predict the likelihood of heart disease using **machine learning techniques**. It mainly uses two models — **LightGBM** and **Random Forest** — and compares their performance to find which one gives better and faster results. The main goal of this project is to help in the **early detection of heart disease** based on medical data such as age, gender, blood pressure, cholesterol, heart rate, and other health indicators.

The dataset used in this project contains several important health-related features that influence heart disease. The data is first preprocessed by checking for missing values, scaling the features using **StandardScaler**, and dividing it into training and testing sets. The models are then trained on this data to learn patterns that can help identify whether a person is likely to have heart disease or not.

After training, both models are evaluated based on **accuracy**, **confusion matrix**, and **classification report**. Their **training times** are also compared to see which algorithm is more efficient. The **LightGBM model** performs slightly better in terms of both accuracy and speed, while the **Random Forest model** also gives stable and reliable results.

The project includes several **visualizations** using **Matplotlib** and **Seaborn** to make the analysis more clear and interesting. These include plots such as the **ROC Curve**, **Confusion Matrix**, **Feature Importance Chart**, **Age Distribution**, and **Correlation Heatmap**. These visual graphs help in understanding which features have the most impact on predicting heart disease.

Another highlight of this project is the **interactive prediction system**. Users can enter their personal health details like age, chest pain type, cholesterol level, and more through the console, and the system will calculate the **probability of heart disease**. Based on this probability, it classifies the user’s health condition into **Low Risk**, **Medium Risk**, or **High Risk** categories. This makes the model easy to use and practical for demonstration or learning purposes.

In conclusion, this project shows how machine learning can be used effectively for **healthcare prediction systems**. By comparing LightGBM and Random Forest, it helps understand which algorithm works best for medical data analysis. In the future, this project can be expanded into a **web application** using tools like **Flask** or **Streamlit**, allowing users to easily access predictions through a website. It’s a great example of how data science and AI can be used to support early diagnosis and spread awareness about heart disease.

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- LightGBM  

---

## 🧠 Models Compared
- **LightGBM Classifier**
- **Random Forest Classifier**

---

## 📈 Key Insights
- LightGBM gives slightly higher accuracy and faster training time.  
- Visualizations help in understanding the relationship between health factors and disease risk.  
- Real-time prediction allows users to estimate their heart disease probability easily.  

---

## 💡 Future Scope
This project can be further improved by adding:
- A **web interface** using Flask or Streamlit  
- **Explainable AI tools** like SHAP or LIME  
- A **PDF report generator** for health summaries  
- **Deployment** on Streamlit Cloud or Hugging Face Spaces  

