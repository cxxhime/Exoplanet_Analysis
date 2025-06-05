# Exoplanet Classification and Habitability Analysis (NASA Exoplanet Archive)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Exoplanet Analysis Project

## Project Objective
Comprehensive analysis of 5,000+ confirmed exoplanets from NASA Exoplanet Archive to identify classification patterns, validate physical relationships, and discover potentially habitable candidates using statistical analysis and machine learning techniques.

## Key Results
- **Habitability Analysis**: 4 potentially habitable candidates identified from 1,639 analyzed (0.2% success rate)
- **Classification Performance**: 96.3% accuracy in predicting detection methods using planetary characteristics (KNN & SVM)
- **Physical Validation**: Kepler's third law confirmed with 99.9% conformity across the dataset
- **Clustering Analysis**: 4 distinct planetary groups discovered through unsupervised learning
- **Density Classification**: 391 gaseous vs 209 rocky planets identified using ML-enhanced approach
- **Comparative Study**: ML approach (86.1% precision) vs simple threshold (83.7% agreement) for composition classification

## Dataset
- **Source**: NASA Exoplanet Archive
- **Size**: 5,903 confirmed exoplanets (after cleaning)
- **Key Variables**: Mass, radius, orbital period, equilibrium temperature, stellar parameters, discovery method
- **Data Quality**: Comprehensive cleaning with strategic NaN handling and 95th percentile outlier filtering

## Methodology

### 1. Exploratory Data Analysis
- Univariate and bivariate statistical analysis with log-scale transformations
- Correlation matrix analysis revealing key physical relationships (stellar mass ↔ temperature: 0.46)
- Distribution analysis identifying observational biases toward massive, close-in planets
- Strategic data filtering to preserve maximum information while ensuring analysis quality

### 2. Physical Law Validation
- **Kepler's third law verification**: T²/a³ = 4π²/G(M+m)
- Quantitative validation with mean difference ≈ 2.25e-19
- Unit conversions: days→seconds, AU→meters, Earth/Jupiter masses→kg
- Outlier identification confirming measurement errors rather than physics violations

### 3. Classification Systems
- **Size Classification**: 8-category system (Mars-sized to Super-Jupiter-sized) based on radius
- **Density Classification**: Binary rocky/gaseous using 3.0 g/cm³ threshold derived from solar system data
- **Temporal Analysis**: Previously verified vs newly verified Kepler planets
- **Comparative Study**: Threshold-based vs Logistic Regression approaches

### 4. Machine Learning Implementation
- **Unsupervised Learning**: K-Means clustering (k=4 optimal via elbow method)
- **Supervised Learning**: Multi-algorithm comparison (KNN: 96.3%, SVM: 96.3%, Logistic Regression: 86.1%)
- **Feature Engineering**: StandardScaler normalization, strategic feature selection
- **Model Validation**: Train/test split (70/30) with proper data preprocessing

### 5. Habitability Assessment
- **Criteria Definition**: Temperature (200-350K), radius (0.5-2 R_Earth), energy flux (0.8-1.2x Earth)
- **Training Data**: NASA exoplanet dataset with binary habitable/non-habitable classification
- **Predictive Modeling**: Logistic Regression achieving 100% accuracy on test set
- **Solar System Validation**: Post-training test on separately created solar system dataset
- **Validation Results**: Earth correctly identified as habitable, gas giants correctly classified as non-habitable

## Key Visualizations

### Physical Relationships
![image](https://github.com/user-attachments/assets/9b0f4284-4eec-4bcd-9be3-2c1228e4ef59)
*Correlation matrix heatmap revealing key relationships between planetary and stellar parameters*

### Clustering Analysis  
![image](https://github.com/user-attachments/assets/4c91536e-342b-41da-9ded-9798929177cf)
*Four distinct planetary groups identified through K-Means unsupervised learning*

### Classification Results
![image](https://github.com/user-attachments/assets/3e2840be-9b1d-4d3c-8663-d387147afc5a)
*Density-based classification showing clear separation between rocky and gaseous populations*

## Technical Stack
```
pandas>=1.3.0          
numpy>=1.21.0         
scikit-learn>=1.0.0   
matplotlib>=3.4.0     
seaborn>=0.11.0       
```

## Project Structure
```
exoplanet-analysis/
├── Analysis/
│   ├── Analysis.ipynb         # Complete analysis pipeline
│   └── dataset_cleaning.ipynb # Data preprocessing
├── Datasets/
│   ├── Exoplanets.csv         # Raw NASA data
│   └── Exoplanets_cleaned.csv # Processed dataset
├── LICENSE.md
└── README.md
```

## Installation and Usage

### Prerequisites
```bash
git clone https://github.com/cxxhime/Exoplanet_Analysis.git
cd Exoplanet_Analysis
pip install -r requirements.txt
```

### Execution
```bash
# Complete analysis pipeline
jupyter notebook Analysis/Analysis.ipynb

# Data cleaning process  
jupyter notebook Analysis/dataset_cleaning.ipynb
```

## Detailed Results

### Classification Outcomes
- **Size Distribution**: Super-Earth and Sub-Neptune categories dominate (observational bias confirmed)
- **Composition Analysis**: ML approach identifies 30 additional rocky planets compared to simple threshold
- **Detection Method**: Transit method dominance confirmed with 96.3% prediction accuracy
- **Temporal Patterns**: Kepler mission contributions clearly visible in discovery timeline

### Identified Habitable Candidates
1. **Kepler-1544 b** - Temperature and size within habitable criteria
2. **Kepler-155 c** - Optimal energy flux and radius conditions  
3. **Kepler-296 e** - Earth-like temperature and size parameters
4. **LP 890-9 c** - Recently discovered candidate in habitable zone

### Model Performance Metrics
- **K-Means Clustering**: 4 optimal clusters with clear physical interpretation
- **Detection Method Classification**: KNN and SVM achieve 96.3% accuracy
- **Composition Classification**: Logistic Regression 86.1% vs threshold 83.7%
- **Habitability Prediction**: 100% precision on exoplanet test set, validated independently on solar system

## Scientific Insights

### Observational Biases Confirmed
- **~50% gaseous planets detected**: Reflects detection method bias toward large, transit-visible planets
- **Hot Jupiters overrepresented**: Close-in massive planets easier to detect via radial velocity
- **Earth-like planets underrepresented**: Current technology limitations for small, distant planets

### Physical Relationships Validated
- **Stellar properties correlation**: Massive stars are hotter (r=0.46)
- **Planet-star energy transfer**: Stellar temperature affects planetary equilibrium temperature (r=0.42)
- **Kepler's law universality**: 99.9% conformity validates gravitational physics across systems

## Limitations and Future Work

### Identified Limitations
- **Observational bias** toward massive, close-in planets due to detection method constraints
- **Missing data handling**: 60-80% missing values in some critical parameters
- **Earth-centric habitability criteria**: Limited to temperature, size, and energy flux
- **Static analysis**: No consideration of stellar variability or atmospheric composition

### Proposed Improvements
- **Integration of JWST data**: Atmospheric composition analysis for refined habitability assessment
- **Expanded habitability criteria**: Include magnetic field, atmospheric pressure, stellar stability
- **Time-series analysis**: Stellar variability and planetary transit timing variations
- **Advanced imputation**: ML-based missing data reconstruction techniques

## Business Applications

### Data Science Competencies Demonstrated
- **Large-scale data processing**: Handling 5,000+ observations with complex missing data patterns
- **Statistical hypothesis testing**: Kepler's law validation with quantitative assessment
- **Machine learning pipeline**: End-to-end supervised and unsupervised learning implementation
- **Model validation**: Proper train/test methodology with independent solar system validation
- **Scientific communication**: Clear visualization and interpretation of complex astrophysical data

### Transferable Skills
- **Anomaly detection**: Outlier identification applicable to fraud detection and quality control
- **Classification systems**: Multi-class prediction transferable to customer segmentation
- **Feature engineering**: Handling diverse data types and scales relevant to financial modeling
- **Validation methodology**: Robust testing approaches applicable to any predictive modeling context

## Citation

If you use this work, please cite:
```
Exoplanet Classification and Habitability Analysis - NASA Archive Study
Nana CHEN, 2025
Data Science Project - GitHub: https://github.com/cxxhime/Exoplanet_Analysis
```

*This project demonstrates advanced data science capabilities through real-world scientific analysis, combining domain expertise with technical implementation for actionable insights.*

## Contact

**Nana CHEN**  
Data Science Student | Aspiring Banking Data Analyst <br>
📧 [Email](mailto:caroline.chen1801@gmail.com) 
🔗 [LinkedIn](https://linkedin.com/in/cxxhime)
