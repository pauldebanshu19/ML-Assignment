# Machine Learning Assignments Collection

A comprehensive collection of machine learning projects covering various algorithms and techniques, from basic regression to advanced clustering methods.

## 📚 Table of Contents

### Regression Models
- **Assignment 1a**: Linear Regression - CO2 Emission Prediction
- **Assignment 1b**: Linear Regression - Car Price Prediction  
- **Assignment 2**: Multiple Linear Regression - Housing Price Prediction
- **Assignment 3**: Linear Regression with Gradient Descent - Salary Prediction
- **Assignment 4**: Non-Linear Regression - China GDP Prediction (1960-2014)

### Classification Models
- **Assignment 5**: Logistic Regression - Cancer Classification (Benign vs Malignant)
- **Assignment 6**: K-Nearest Neighbors (KNN) - Telecommunications Customer Classification
- **Assignment 7**: Decision Tree - Drug Classification
- **Assignment 8**: Support Vector Machine (SVM) - Diabetes Prediction
- **Assignment 9**: Support Vector Machine (SVM) - Cancer Classification

### Neural Networks
- **Assignment 10**: Neural Network - Diabetes Prediction
- **Assignment 11**: Neural Network - Advanced Implementation

### Clustering
- **Assignment 12**: K-Means Clustering - Iris Dataset
- **Assignment 13**: K-Means Clustering - Customer Segmentation
- **Assignment 14**: Hierarchical Clustering - Vehicle Classification
- **Assignment 15**: Hierarchical Clustering - Weather Station Analysis

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - pandas - Data manipulation
  - numpy - Numerical computing
  - matplotlib & seaborn - Data visualization
  - scikit-learn - Machine learning algorithms
  - TensorFlow/Keras - Neural networks (Assignments 10-11)

## 📊 Key Features

- **Comprehensive Coverage**: From basic linear regression to advanced neural networks
- **Real-World Datasets**: Housing prices, medical data, customer data, GDP trends
- **Model Evaluation**: Accuracy, precision, recall, F1-score, R² score, confusion matrices
- **Visualization**: Scatter plots, ROC curves, decision boundaries, dendrograms
- **Best Practices**: Data preprocessing, feature scaling, train-test splits, hyperparameter tuning

## 🚀 Getting Started

### Prerequisites

**Python 3.8+** is required. All dependencies are listed in `requirements.txt`.

### Installation

1. **Clone this repository**
   ```bash
   git clone <repository-url>
   cd ML-Assignment
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Navigate to any assignment folder** (e.g., `Assignment 1/Assignment a/`)

3. **Open the `model.ipynb` file**

4. **Run all cells** to see the complete analysis and results

### Quick Start Example
```bash
# After installation
cd "Assignment 1/Assignment a"
jupyter notebook model.ipynb
```

## 📈 Assignment Highlights

### Regression
- **Linear Regression**: Predicting continuous values (prices, salaries, emissions)
- **Non-Linear Regression**: Polynomial features for complex relationships
- **Gradient Descent**: Custom implementation from scratch

### Classification
- **Logistic Regression**: Binary classification with probability estimates
- **KNN**: Instance-based learning with optimal k selection
- **Decision Trees**: Rule-based classification with visualization
- **SVM**: Maximum margin classification with kernel tricks

### Neural Networks
- **Deep Learning**: Multi-layer perceptrons for complex patterns
- **Activation Functions**: ReLU, sigmoid for non-linearity
- **Optimization**: Adam optimizer, dropout for regularization

### Clustering
- **K-Means**: Partitioning data into k clusters
- **Hierarchical**: Dendrogram-based clustering
- **Applications**: Customer segmentation, pattern discovery

## � Paroject Structure

```
ML-Assignment/
├── Assignment 1/
│   ├── Assignment a/
│   │   ├── model.ipynb
│   │   └── fuel_consumption_dataset.csv
│   └── Assignment b/
│       ├── model.ipynb
│       └── used_cars_dataset.csv
├── Assignment 2/
│   ├── model.ipynb
│   └── housing_price_dataset.csv
├── Assignment 3/
│   ├── model.ipynb
│   └── salary_data.csv
├── Assignment 4/
│   ├── model.ipynb
│   └── china_gdp.csv
├── Assignment 5-15/
│   └── ... (similar structure)
├── requirements.txt
├── .gitignore
└── README.md
```

## 📝 Dataset Information

Each assignment includes its own dataset:
- `fuel_consumption_dataset.csv` - Vehicle emissions data
- `used_cars_dataset.csv` - Car pricing data
- `housing_price_dataset.csv` - Real estate data
- `salary_data.csv` - Employee compensation data
- `china_gdp.csv` - Economic indicators
- `samples_cancer.csv` - Medical diagnostic data
- `teleCust.csv` - Telecommunications customer data
- `drug.csv` - Pharmaceutical data
- `pima-indians-diabetes.data.csv` - Diabetes indicators
- `iris.csv` - Classic iris flower dataset
- `Cust_Segmentation.csv` - Customer behavior data
- `Vehicle.csv` - Vehicle characteristics
- `WeatherStation.csv` - Meteorological data

## 🎯 Learning Outcomes

- Understanding of supervised and unsupervised learning
- Hands-on experience with scikit-learn and TensorFlow
- Model evaluation and performance metrics
- Data preprocessing and feature engineering
- Hyperparameter tuning and model optimization
- Visualization of machine learning results

## � CDevelopment Setup

### Virtual Environment
This project uses a virtual environment to manage dependencies. The `.gitignore` file is configured to exclude:
- Virtual environment folders (`venv/`, `env/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Jupyter checkpoints (`.ipynb_checkpoints/`)
- IDE configuration files (`.vscode/`, `.idea/`)
- OS-specific files (`.DS_Store`, `Thumbs.db`)

### Dependencies
All required packages are specified in `requirements.txt`:
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Deep Learning**: tensorflow, keras
- **Notebook Environment**: jupyter, ipykernel

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt
```

**Issue**: Jupyter kernel not found
```bash
# Solution: Install ipykernel in your virtual environment
python -m ipykernel install --user --name=venv
```

**Issue**: TensorFlow installation problems
```bash
# Solution: Install TensorFlow separately
pip install tensorflow==2.10.0
```

## 📊 Performance Metrics Summary

| Assignment | Algorithm | Dataset | Accuracy/R² |
|-----------|-----------|---------|-------------|
| 1a | Linear Regression | CO2 Emissions | R² > 0.85 |
| 1b | Linear Regression | Car Prices | R² > 0.52 |
| 2 | Multiple Linear Reg. | Housing | R² > 0.91 |
| 5 | Logistic Regression | Cancer | Acc > 95% |
| 6-7 | KNN/Decision Tree | Telecom/Drug | Acc > 85% |
| 8-9 | SVM | Diabetes/Cancer | Acc > 75% |
| 12-15 | Clustering | Various | Silhouette > 0.5 |

## 📧 Contact

For questions or suggestions about these assignments, feel free to reach out!

## 📄 License

This project is for educational purposes.

---

**Note**: All assignments are implemented in Jupyter Notebooks with detailed explanations, visualizations, and performance metrics. Each notebook is self-contained and can be run independently.
