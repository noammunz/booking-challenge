# Kaggle Recommender System Competition

This repository contains code for a Kaggle recommender system competition, where the task is to match a user with the review they wrote for a specific booking from all available reviews for that accommodation.

## Files

- **`create_datasets.ipynb`**: Creates and saves objects for dataset creation during training.
- **`inference.ipynb`**: Performs inference using an ensemble of LightGBM models and generates a submission file.
- **`Kmeans.ipynb`**: Clusters instances into similar groups to aid contrastive loss in learning meaningful differences.
- **`preprocess.ipynb`**: Preprocesses data for users and reviews, including tokenization and processing.
- **`train_embedder.ipynb`**: Trains embedding models used before the boosting model.
- **`train_lgbm.ipynb`**: Trains a LightGBM model using the embedding model and creates hard negatives.

## Requirements

- Python 3.x
- pandas, numpy, lightgbm, sklearn, etc.

## Usage

1. Preprocess the data using `preprocess.ipynb`.
2. Create datasets with `create_datasets.ipynb`.
3. Train the embedding model with `train_embedder.ipynb`.
4. Train the LightGBM model with `train_lgbm.ipynb`.
5. Perform inference and generate the submission file with `inference.ipynb`.
6. Use K-means clustering for grouping in `Kmeans.ipynb`.

## Evaluation Metrics

The model performance in this competition is evaluated using the **Mean Reciprocal Rank at 10 (MRR@10)** metric. This metric measures how well the top-10 recommendations rank the correct review for a given user. Higher values indicate better performance, with a perfect score of 1.0 meaning the correct review is always ranked in the top 10.

### MRR@10 Calculation
MRR@10 is calculated as:
\[
MRR@10 = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
\]
Where:
- \(Q\) is the set of all query-user pairs.
- \(\text{rank}_i\) is the rank position of the first relevant review for the \(i^{th}\) user in the top-10 list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
