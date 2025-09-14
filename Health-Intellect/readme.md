# Health-Intellect: AI-Powered Diabetes Risk Advisor 🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3-blue?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end data science project that predicts the risk of diabetes based on health indicators. The project encompasses the entire machine learning lifecycle, from in-depth exploratory data analysis (EDA) and feature engineering to model training, tuning, and deployment as an interactive web application.


---

### 📖 About The Project

This project was built to demonstrate a comprehensive understanding of the data science workflow. Using the **Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset**, this application provides a user-friendly interface to predict a user's likelihood of having diabetes.

The core of the project is a highly accurate **LightGBM classification model**, which was selected after a rigorous evaluation against other algorithms, achieving a final **ROC-AUC score of 0.83+**. The entire application is built with a modern, responsive user interface using **Flask**.

---

### 🚀 Key Features

-   **In-Depth EDA:** Comprehensive exploratory data analysis to uncover patterns and correlations in the data.
-   **Feature Engineering:** Created new, insightful features like `HealthScore` and `UnhealthyDays` to improve model performance.
-   **Machine Learning Modeling:** Trained and evaluated multiple models, selecting a tuned LightGBM classifier as the final model.
-   **Interactive Web UI:** A clean, modern, and fully responsive web application built with Flask for users to input their health data.
-   **Real-time Prediction:** The application provides an instant probability score, classifying the risk as low, medium, or high.

---

### 🛠️ Tech Stack

-   **Backend:** Flask
-   **Data Science:** Pandas, NumPy, Scikit-learn, LightGBM
-   **Frontend:** HTML5, CSS3
-   **Deployment:** Vercel 

---

### 🔧 Getting Started

To get a local copy up and running, follow these simple steps.

#### Prerequisites

-   Python 3.8+
-   Git

#### Installation & Setup

1.  **Create a `requirements.txt` file:**
    Before you begin, make sure you are in your project's main directory and have your virtual environment activated. Then, run:
    ```sh
    pip freeze > requirements.txt
    ```

2.  **Clone the repository:**
    ```sh
    git clone [https://github.com/](https://github.com/)thekushak0003/[Health-Analytics].git
    ```

3.  **Navigate to the project directory:**
    ```sh
    cd Health-Intellect
    ```

4.  **Create and activate a virtual environment:**
    ```sh
    # Create the environment
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

5.  **Install the required libraries:**
    ```sh
    pip install -r requirements.txt
    ```

### How to Run

1.  With your virtual environment activated, run the Flask application:
    ```sh
    python app.py
    ```
2.  Open your web browser and navigate to `http://1227.0.0.1:5000` to see the application in action.

---

---

### 📄 License

This project is distributed under the MIT License.

---

### 👤 Contact

**Abhay Tiwari**

-   **LinkedIn:** [Abhay Tiwari](https://www.linkedin.com/in/thekushak/)
-   **Email:** [abhay4contact@gmail.com](mailto:abhay4contact@gmail.com)
