# 💊 PharmaAssist: A Drug Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Methodology](#-methodology)
- [Application Showcase](#-application-showcase)
- [How to Run This Project Locally](#-how-to-run-this-project-locally)
- [Future Scope](#-future-scope)
- [About the Author](#-about-the-author)

---

## 🚀 Project Overview

### Problem Statement
In a clinical setting, doctors and pharmacists often need to find substitutes for a specific drug due to unavailability, patient allergies, or contraindications. Manually searching for alternatives with similar therapeutic effects and manageable side effects can be time-consuming and prone to error.

### Solution
PharmaAssist provides a data-driven solution by leveraging a **content-based filtering** model. A user can select a drug, and the system instantly recommends the top 5 most similar drugs by analyzing a "feature soup" of their properties. The model was built to be memory-efficient to handle a large dataset of over 200,000 medicines.

### Key Features
-   **Intelligent Recommendations**: Suggests drug substitutes based on a holistic profile, not just active ingredients.
-   **Memory-Efficient Design**: Calculates similarity scores on-the-fly, avoiding memory errors common with large datasets.
-   **Interactive User Interface**: A clean and intuitive web application built with Streamlit for easy use.
-   **In-Depth Data Analysis**: The project is built on a foundation of rigorous Exploratory Data Analysis (EDA) to clean the data and uncover meaningful patterns.

### Tech Stack
-   **Language**: Python
-   **Libraries**:
    -   Data Manipulation: Pandas, NumPy
    -   Machine Learning: Scikit-learn
    -   Web Framework: Streamlit
    -   Visualization (for EDA): Matplotlib, Seaborn, Plotly

---

## 🔬 Methodology

### 1. Data Source
The model was trained on the [250k Medicines Usage, Side Effects, and Substitutes dataset](https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes) from Kaggle.

### 2. Data Cleaning & EDA
The initial dataset was messy, with information spread across 59 columns. The first step was a rigorous cleaning and EDA phase:
-   **Consolidated Columns**: Multiple columns for `sideEffect` and `use` were merged into single, analyzable columns.
-   **Handled Missing Values**: Rows with critical missing data (like `Therapeutic Class`) were dropped, while others were filled with meaningful placeholders like "Unknown" or "None Reported."
-   **Uncovered Insights**: EDA revealed key patterns, such as the high prevalence of habit-forming drugs within the "ANALGESICS & ANTIPYRETICS" class, which informed the model's feature design.

### 3. Modeling Approach
The core of PharmaAssist is a content-based filtering model.
-   **Feature Engineering**: A "feature soup" was created for each drug by concatenating its most important textual properties: `Therapeutic Class`, `Action Class`, `Chemical Class`, `all_side_effects`, and `all_uses`.
-   **TF-IDF Vectorization**: The text "soup" for each drug was converted into a numerical vector using `TfidfVectorizer`. This technique represents the importance of a word in a drug's profile relative to the entire dataset.
-   **Cosine Similarity**: To find the "similarity" between two drugs, the cosine of the angle between their TF-IDF vectors was calculated. A score closer to 1 indicates higher similarity. To handle the large dataset without causing a `MemoryError`, the `linear_kernel` was computed on-the-fly for a single selected drug against the entire dataset, rather than pre-computing the entire similarity matrix.

---

## 🖥️ Application Showcase

The final model is deployed in an interactive Streamlit application. The UI allows users to:
1.  Select a drug from a dropdown list with a search-as-you-type feature.
2.  View the top 5 substitutes displayed in a clean, card-based layout, showing key information like primary uses, therapeutic class, and habit-forming status.

---

## 🛠️ How to Run This Project Locally

Follow these steps to set up and run the project on your own machine.

### Prerequisites
-   Python 3.9 or higher
-   pip package manager

### 1. Clone the Repository
```bash
git clone [https://github.com/your-github-username/pharma-assist.git](https://github.com/your-github-username/pharma-assist.git)
cd pharma-assist
