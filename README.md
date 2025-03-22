# Credit Card Approval Prediction

## Project Overview
This project focuses on predicting credit card approvals using machine learning techniques. By analyzing applicant data and credit history, the model aims to identify which applicants are likely to be approved for credit cards, helping financial institutions make more informed decisions.

The final model is a tuned stacked ensemble that achieves high precision while maintaining good recall on the minority class, making it well-suited for the class imbalance typically found in credit approval datasets.

## Data Source
The dataset used in this project is available on Kaggle:
[Credit Card Approval Prediction Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

The data consists of two main files:
- `application_record.csv`: Contains applicant demographic information
- `credit_record.csv`: Contains applicant credit history

**Note:** Due to GitHub file size limitations, raw data files are not included in this repository. Please download them from the Kaggle link above.

## Project Structure
```
├── Data/
│   ├── processed/       # Cleaned and processed data files
│   ├── engineered/      # Feature engineered data
│   └── data_description.txt  # Description of data fields
├── Models/
│   ├── stacked_model_with_threshold.joblib  # Saved model file
│   ├── model_metadata.json                  # Model configuration details
│   └── model_performance.csv                # Model performance metrics
└── Notebooks/
    ├── 01_Data_Wrangling.ipynb             # Data cleaning and preprocessing
    ├── 02_Exploratory_Data_Analysis.ipynb  # EDA and insights
    ├── 03_Feature_Engineering.ipynb        # Feature creation and selection
    └── 04_Model_Building.ipynb             # Model development and evaluation
```

## Model Performance
The final model is a tuned stacked ensemble with the following performance metrics:

| Metric | Value |
|--------|-------|
| Accuracy | 0.6726 |
| Precision | 0.9942 |
| Recall | 0.6709 |
| F1 Score | 0.8012 |
| Minority Class Recall | 0.7730 |
| Business Metric | 0.7871 |

The model uses a custom threshold of 0.67 to optimize for business value, balancing the cost of false approvals against missed opportunities.

## Key Features
- **Stacked ensemble architecture**: Combines multiple base models to improve overall performance
- **Class imbalance handling**: Special techniques to address the imbalanced nature of credit approval data
- **Threshold optimization**: Custom threshold tuning to maximize business value rather than just statistical metrics
- **Comprehensive evaluation**: Multiple metrics assessed to ensure model robustness

## Setup and Installation

### Prerequisites
- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  xgboost
  lightgbm
  joblib
  imbalanced-learn
  ```

### Installation
1. Clone this repository:
```
git clone https://github.com/sgessesse/credit-card-approval-predictions.git
cd credit-card-approval-predictions
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Download the data from Kaggle and place the files in the `Data/` directory.

## Running the Project
The notebooks should be run in numerical order:

1. **Data Wrangling**: Cleans and processes the raw data
2. **Exploratory Data Analysis**: Explores patterns and relationships in the data
3. **Feature Engineering**: Creates and selects important features
4. **Model Building**: Builds, tunes, and evaluates models

## License
[MIT License](LICENSE)

## Acknowledgments
- The dataset is provided by Kaggle user rikdifos
- This project was completed as part of a data science portfolio development 