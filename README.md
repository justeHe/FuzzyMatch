# Data Mining Assignment 1 - Name Matching and Deduplication

## Project Description
This project implements fuzzy name matching and deduplication functionality, including:

1. Name fuzzy matching using fuzzywuzzy and rapidfuzz libraries
2. Multi-threaded parallel matching for improved efficiency
3. Accuracy evaluation of matching results
4. Deduplication of raw data

## File Description

- `data_matcher.py`: Main program file containing matching and deduplication logic
- `primary.csv`: Original primary name data
- `alternate.csv`: Original alternate name data
- `test_01.csv`: Test data
- `matched_results.csv`: Matching results output
- `deduplicated_results.csv`: Deduplication results output

## Usage

1. Install required dependencies:
```
pip install pandas fuzzywuzzy rapidfuzz tqdm
```

2. Run `HW1.ipynb` file

3. Results will be saved to `matched_results.csv` and `deduplicated_results.csv`

## Matching Example
```
39777,"SRABIONOV, T K",39777,"SRABIONOV, Tigran Khristoforovich",85.71428571428571,1
```

## Author
Hedong