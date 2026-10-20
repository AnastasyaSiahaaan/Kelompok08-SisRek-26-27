# GABIN — Recommender System Project

**Working Title:** The Impact of User Interaction History on Collaborative, Content-Based, and Hybrid Recommendation Performance

## 1. Project Overview
A recommender system project that investigates the relationship between user interaction history and recommendation performance. 

The project compares four recommendation approaches: Popularity-Based Recommendation, Matrix Factorization (MF), Content-Based Filtering (CBF), and Hybrid Recommendation.

The evaluation focuses on three user activity groups based on the number of interactions in the training data: Low, Medium, and High. The objective is to examine how recommendation performance differs across these groups.

## 2. Objectives
The objectives of this project are:
- To implement a popularity-based recommendation baseline.
- To implement Matrix Factorization using TruncatedSVD.
- To implement Content-Based Filtering using TF-IDF and item metadata.
- To implement a hybrid recommender that combines MF and CBF scores.
- To evaluate recommendation performance using Recall@10 and NDCG@10.
- To compare overall performance and performance across user activity groups.

## 3. Recommendation Methods
### 3.1 Popularity-Based Recommendation
This baseline ranks candidate items according to the number of positive training interactions. An interaction is considered positive when its rating is at least 4. Items with equal popularity are ordered by item ID.

### 3.2 Matrix Factorization (MF)
The MF baseline uses TruncatedSVD to obtain a latent-factor approximation of the user-item interaction matrix. The resulting scores are used to rank candidate items for each user.

### 3.3 Content-Based Filtering (CBF)
The CBF baseline uses TF-IDF to represent item metadata, including available title, category, feature, description, and content-text fields. User profiles are constructed from positively rated items, and cosine similarity is used to score candidate items.

### 3.4 Hybrid Recommendation
The hybrid baseline combines the MF and CBF scores. Scores are normalized for each user using min-max normalization, and both methods receive equal weights with alpha = 0.5.

For the MF, CBF, and hybrid recommendation methods, items already observed in the user's training history are excluded from the recommendation list.

## 4. Dataset
The project uses the provided interaction dataset and supporting evaluation files.

| Dataset | Number of records |
|---|---:|
| Training interactions | 427,957 |
| Full validation interactions | 41,284 |
| Warm validation interactions | 30,360 |
| Test interactions | 42,595 |
| Candidate items | 22,753 |
| Users in warm validation evaluation list | 11,659 |

The warm validation data is used for the current baseline evaluation. The full validation set contains additional interactions beyond the warm validation subset. The test set is reserved for final evaluation and is not used to select or tune the current baselines.

The candidate item file defines the items considered for recommendation. The popularity ranking is checked to ensure that all candidate items are represented.

## 5. User Activity Groups
Users are grouped according to their number of interactions in the training data:

| Activity group | Number of training interactions |
|---|---|
| Low | 1–5 |
| Medium | 6–8 |
| High | 9 or more |

These groups are used to compare recommendation performance for users with different amounts of interaction history.

## 6. Evaluation Metrics
### 6.1 Recall@10
Recall@10 measures the proportion of relevant items retrieved among the top 10 recommendations. An item is considered relevant when its validation rating is at least 4.

### 6.2 NDCG@10
NDCG@10 evaluates the quality of the top 10 recommendation ranking by considering the positions of relevant items. Relevant items ranked higher contribute more to the score.

Both metrics are calculated overall and separately for each user activity group.

## 7. Project Structure
```text
Kelompok08-SisRek-26-27/
├── data/
│   ├── raw/
│   │   ├── interactions_train.csv
│   │   ├── candidate_items_warm.csv
│   │   └── items.csv
│   └── supporting/
│       ├── interactions_validation_warm.csv
│       └── evaluable_users_validation_warm.csv
├── src/
│   ├── baseline_common.py
│   ├── popularity_baseline.py
│   ├── evaluate_popularity.py
│   ├── evaluate_popularity_by_group.py
│   ├── mf_baseline.py
│   ├── tfidf_cbf_baseline.py
│   ├── hybrid_baseline.py
│   ├── compare_results.py
│   ├── check_evaluation.py
│   ├── check_popularity.py
│   ├── check_recommendations.py
│   ├── check_user_groups.py
│   ├── inspect_data.py
│   ├── inspect_supporting.py
│   └── README.md
├── results/
├── requirements.txt
└── README.md
```

The `src/` directory contains the baseline implementations, evaluation scripts, comparison utilities, validation checks, and data inspection scripts. The `results/` directory stores the generated evaluation outputs.

## 8. Environment Setup
The project is developed using Python and can be run from the VS Code terminal on Windows.

Activate the project's virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Run the commands from the project root directory.

## 9. Running the Experiments
Run the following scripts in order.

### 9.1 Popularity Baseline
```powershell
python src/popularity_baseline.py
```

### 9.2 Matrix Factorization Baseline
```powershell
python src/mf_baseline.py
```

### 9.3 Content-Based Filtering Baseline
```powershell
python src/tfidf_cbf_baseline.py
```

### 9.4 Hybrid Baseline
```powershell
python src/hybrid_baseline.py
```

### 9.5 Validate Recommendation Outputs
```powershell
python src/check_recommendations.py
python src/check_popularity.py
```

### 9.6 Compare Results
```powershell
python src/compare_results.py
```

The evaluation scripts save their outputs in the `results/` directory.

## 10. Baseline Results
The current warm-validation results are:

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0.01350 | 0.00922 |
| MF (TruncatedSVD) | 0.01309 | 0.00792 |
| TF-IDF CBF | 0.01257 | 0.00730 |
| Hybrid | 0.01457 | 0.00917 |

The hybrid model achieves the highest overall Recall@10 among the four evaluated methods. The popularity baseline achieves the highest overall NDCG@10, slightly exceeding the hybrid model.

These results are descriptive comparisons on the current validation evaluation. They do not establish statistical significance or guarantee that the same ranking will hold on the test set.

## 11. Output Validation
The recommendation output checks verify that:
- The MF, CBF, and hybrid recommendation lists contain no duplicate recommendations.
- Each evaluated user receives exactly 10 recommendations.
- Recommended items belong to the candidate item set.
- Items already observed in the training history are excluded from MF, CBF, and hybrid recommendation lists.
- The popularity ranking includes all 22,753 candidate items.

The validation scripts should be rerun whenever the recommendation-generation code changes.

## 12. Limitations and Next Steps
The current results represent baseline experiments on the warm validation subset. Further work should examine performance differences across user activity groups, review the strengths and weaknesses of each recommendation method, and conduct the final evaluation using the reserved test data.

Any additional tuning should be performed without using the test set to select model settings. Future conclusions should be based on the actual evaluation outputs rather than assumptions about which method will perform best.

## 13. Team and Contributions
This project is developed by Group G08 for the Recommender Systems course.

Individual contributions, code changes, and experiment responsibilities should be documented by the team based on the work actually completed by each member.

## 14. AI Usage Declaration
Any AI tools used during the project should be declared according to the course requirements. The declaration should accurately describe how the tools were used, such as for understanding concepts, debugging, reviewing documentation, or improving written explanations. All code, experimental outputs, and conclusions should be checked by the project team.