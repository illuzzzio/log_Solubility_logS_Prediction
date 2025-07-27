🧪 Molecular Solubility Prediction using Machine Learning

This project uses machine learning to predict the logarithmic solubility (logS) of molecules from their structural descriptors using two models: Linear Regression and Random Forest Regressor. The entire implementation is contained within a Jupyter Notebook viewable directly on GitHub.
📁 Project Structure

.
├── hypothesis.txt                # Step-by-step methodology explanation
├── visuals_for_ML/              # Folder containing generated plots and visuals
├── Solubility.csv               # Dataset with molecular descriptors
├── machine.ipynb                # Jupyter Notebook with complete implementation
└── README.md                    # You're reading it!

    📌 Note: The code is written in a Jupyter Notebook (machine.ipynb).
    You can preview and read the full code, output, and graphs directly on GitHub.

🎯 Objective

To predict the solubility (logS) of chemical compounds using four key molecular features:

    MolLogP – Lipophilicity

    MolWt – Molecular Weight

    NumRotatableBonds – Flexibility of the molecule

    AromaticProportion – Proportion of aromatic atoms

The target variable (y) is logS.
⚙️ Environment Setup
✅ Step 1: Create and Activate Virtual Environment

# Create virtual environment named 'inv'
python -m venv inv

# Activate it:
# Windows:
inv\Scripts\activate

# macOS/Linux:
source inv/bin/activate

✅ Step 2: Install Required Packages

pip install pandas numpy matplotlib scikit-learn jupyter

💻 Run the Notebook in VS Code

If you're using Visual Studio Code:

    Open the project folder.

    Press Ctrl+Shift+P → Select Python: Select Interpreter → Choose your inv environment.

    Install the Python and Jupyter extensions if not already installed.

    Open machine.ipynb.

    Run the notebook cells interactively.

    You can also open machine.ipynb directly on GitHub to preview the code and results.

📊 Dataset Overview

The file Solubility.csv includes:
Column	Description
MolLogP	LogP (lipophilicity)
MolWt	Molecular weight
NumRotatableBonds	Bond flexibility indicator
AromaticProportion	Ratio of aromatic atoms
logS	Solubility (target variable)
🔁 Project Workflow
1. Data Preparation

    Load Solubility.csv using pandas

    Separate X (first 4 features) and y (logS)

    Visualize feature columns using .drop() method

2. Data Splitting

    Use train_test_split from sklearn.model_selection

    80% training data (915 rows), 20% test data (229 rows)

3. Model 1: Linear Regression

    Fit a linear regression model

    Predict logS on training and test sets

    Evaluate using:

        Mean Squared Error (MSE)

        R² Score (R-squared)

4. Model 2: Random Forest Regressor

    Train a RandomForestRegressor model

    Predict and evaluate performance using the same metrics

    Compare to Linear Regression results

5. Formatting & Visualization

    Use DataFrame.transpose() and .columns for tidying results

    Create graphs and visuals using matplotlib and numpy

    All plots saved in visuals_for_ML/

📈 Sample Visuals
Linear Regression	Random Forest
	

(Replace with actual filenames from your visuals_for_ML/ folder)
📊 Final Comparison
Metric	Linear Regression	Random Forest
MSE	Moderate	Lower
R² Score	Acceptable	Higher
Visual Fit	Less precise	More accurate

🎯 Conclusion: Random Forest Regressor yields better predictions and is more robust for this problem.
🧰 Tech Stack

    Python 3.x

    Jupyter Notebook

    VS Code (recommended IDE)

    Libraries:

        pandas

        numpy

        matplotlib

        scikit-learn

        jupyter

🚀 How to Run This Project

# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name

# Setup virtual environment
python -m venv inv
source inv/bin/activate    # or inv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt    # (optional if you have a requirements file)

# Launch Jupyter Notebook
jupyter notebook machine.ipynb

Or, open machine.ipynb directly in VS Code or GitHub to explore the full implementation.
