# Recommender System Demo using Matrix Factorization (SVD)
This notebook demonstrates a simple movie recommender system built with the Surprise library. It uses matrix factorization (SVD) to predict user–movie ratings and recommend top movies.

## Features

- **User-based Recommendations**: Get personalized movie recommendations for any user
- **Content-based Recommendations**: Find movies similar to a given title
- **Model Evaluation**: Cross-validation and performance metrics (Precision@5, Recall@5, RMSE)
- **Implicit Feedback**: Converts ratings to binary preferences (liked/not liked)

## Dataset

The system uses two CSV files (MovieLens Dataset downloaded form from GroupLens Research Lab - [Link](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)):
- `movies.csv`: Contains movie information (movieId, title, genres)
- `ratings.csv`: Contains user ratings (userId, movieId, rating, timestamp)

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd movie-recommendation-main
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv movie_rec_env

# Activate virtual environment
# On macOS/Linux:
source movie_rec_env/bin/activate
# On Windows:
movie_rec_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**: Navigate to and open `khan-A2-movie_rs.ipynb`

3. **Run all cells**: Execute all cells in order by selecting "Cell" → "Run All" or run each cell individually with `Shift+Enter`

### Running as Python Script

If you prefer to run the code as a Python script:

1. **Convert notebook to Python script**:
   ```bash
   jupyter nbconvert --to script khan-A2-movie_rs.ipynb
   ```

2. **Run the script**:
   ```bash
   python khan-A2-movie_rs.py
   ```

## How It Works

### 1. Data Preprocessing
- Loads movie and rating data
- Converts explicit ratings to implicit feedback (rating ≥ 3.5 = liked)
- Maps users and movies to matrix indices

### 2. Model Training
- Uses LightFM with WARP (Weighted Approximate-Rank Pairwise) loss
- Matrix factorization with 100 latent factors
- Trains for 20 epochs

### 3. Evaluation
- Train/test split (80/20)
- 5-fold cross-validation
- Metrics: Precision@10, Recall@10, F1-score, AUC

### 4. Recommendations

#### Get recommendations for a user:
```python
# Recommend top 5 movies for user 99
recommendations = recommend_by_user(99, 5)
for movie_id, title, score in recommendations:
    print(f"Movie: {title}, Score: {score:.3f}")
```

#### Find similar movies:
```python
# Find movies similar to "Dances with Wolves"
similar_movies = recommend_by_title("Dances with Wolves", 5)
for movie_id, title, similarity in similar_movies:
    print(f"Similar: {title}, Similarity: {similarity:.3f}")
```

## Expected Output

When you run the notebook, you should see:

1. **Model Training Progress**: surprise.model_selection
2. **Evaluation Metrics**: 
   - Precision@5: ~0.6639
   - Recall@5: ~0.4723
   - RMSE: ~0.8817

3. **Sample Recommendations**: Personalized movie recommendations for users 140,603,438
4. **Predict ratings for unseen movies for a specific user**: user 196.


## File Structure

```
movie-recommendation-main/
├── deepak-A2-demo.ipynb    # Main Jupyter notebook
├── movies.csv               # Movie metadata
├── ratings.csv              # User ratings data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Troubleshooting
