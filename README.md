# Record Linkage Project for SCUPI DATAMining

A Python project for data cleaning and matching, developed by the team from Sichuan University-Pittsburgh Institute.

## Project Overview
This project focuses on **record linkage**, aiming to accurately identify and link related records from different data sources. It includes data cleaning, fuzzy matching, and hierarchical matching algorithms.

## Features
- **Data Cleaning**: Standardize and normalize input data using word segmentation and character replacement.
- **Matching Algorithm**: Use `rapidfuzz` for efficient fuzzy matching with multiple scoring methods.
- **Hierarchical Matching**: Three-stage matching process with language detection, dictionary processing, and fuzzy matching.

## Installation
```
pip install -r requirements.txt
```

## Dependencies
- Python >= 3.9
- pandas >= 2.3.0
- rapidfuzz >= 3.13.0
- wordninja

## Results
- **Accuracy**: Up to 96.10% on test datasets
- **Processing Speed**: Parallel processing with ThreadPoolExecutor

## Future Work
- Optimize performance using vectorization and caching.
- Explore advanced algorithms for complex data variations.

## Files
- `data_matcher.py`: Main matching algorithm implementation
- `HW1.ipynb`: Initial experiments and testing
- `HW1_2.ipynb`: Advanced matching techniques and analysis