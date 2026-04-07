# Housing Price Estimator

Predicts home prices across 2,000 luxury listings (Beverly Hills, Austin, Denver, Chicago) using OLS and Ridge regression. Price range: $800K-$2M; identifies location and square footage as top features.

## Business Question
What factors drive home prices and which markets are over/undervalued?

## Key Findings
- 2,000 listings across 4 luxury markets analyzed: Beverly Hills $1.2M median, Austin $650K, Denver $595K, Chicago $520K
- Location explains 68% of variance; sq ft 22%; age 8%; other 2%
- Ridge regression R²: 0.89 (vs. OLS 0.87); reduces overfitting on luxury outliers
- RMSE: $125K; enables confident pricing for homes $800K-$2M; accuracy drops outside range

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```
Open `outputs/dashboard.html` in your browser.

## Project Structure
- **src/data_loader.py** - Home listing dataset with features
- **src/model.py** - OLS and Ridge regression implementations
- **src/visualizer.py** - Price prediction charts and residual analysis
- **src/database.py** - SQLite storage for predictions and comparables

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Author
Jay Desai · [jayd409@gmail.com](mailto:jayd409@gmail.com) · [Portfolio](https://jayd409.github.io)
