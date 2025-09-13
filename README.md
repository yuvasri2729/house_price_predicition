# 🏠 House Price Prediction  

This project is a **Flask web application** that predicts house prices based on user input.  
It also includes **Login and Signup functionality** with a **SQLite database**.  

---

## 📂 Project Structure  

```
house_price_prediction/
│── app.py
│── requirements.txt
│── README.md
│
├── static/
│   ├── style.css
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── predict.html
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   └── train_model.py
│
└── database/
    └── users.db
```

---

## ⚙️ Features  

✅ User **Signup & Login** (with session management)  
✅ **House Price Prediction** using trained ML model (`model.pkl`)  
✅ **SQLite Database** for storing user credentials  
✅ **Flash messages** for login/logout/signup status  
✅ **Simple UI** with HTML + CSS  

---

## 🛠 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/house_price_prediction.git
   cd house_price_prediction
   ```

2. Create a virtual environment (recommended):  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:  
   ```bash
   python app.py
   ```

5. Open your browser and go to:  
   ```
   http://127.0.0.1:5000/
   ```

---

## 🧠 Model  

- The trained ML model is stored in **`models/model.pkl`**  
- You can train a new model using `src/train_model.py` and save it as `model.pkl`  

Example:  

```python
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy training
X = np.array([[2, 1, 1000], [3, 2, 1500], [4, 3, 2000]])
y = np.array([50, 100, 150])

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("models/model.pkl", "wb"))
```

---

## 🔐 Authentication  

- User credentials are stored in **SQLite** database: `database/users.db`  
- Current version stores plain-text passwords (for demo)  
- ⚠️ For production, use **password hashing** (e.g., `werkzeug.security.generate_password_hash`)  

---

## 📸 Screenshots  

- **Home Page**  
- **Signup Page**  
- **Login Page**  
- **Prediction Page**  

(You can add screenshots here later)  

---

## 📌 Future Improvements  

- Add **password hashing** for security  
- Improve UI with **Bootstrap / TailwindCSS**  
- Deploy on **Heroku / Render / AWS**  
- Add more features for prediction (location, year built, etc.)  

---

✍️ **Author:** Your Name  
📅 **Year:** 2025  
