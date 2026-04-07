import pandas as pd
import numpy as np
import os

def load_data():
    """
    Load California Housing dataset. Attempts scikit-learn's cached version,
    falls back to synthetic data if unavailable.
    """
    os.makedirs('data', exist_ok=True)

    # Try sklearn's California Housing
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame.copy()
        df.columns = ['median_income', 'house_age', 'avg_rooms', 'avg_bedrooms',
                      'population', 'avg_occupancy', 'latitude', 'longitude', 'price']
        # Scale price from [0, 5] to actual dollars
        df['price'] = (df['price'] * 100000).round(0)
        df['region'] = df['latitude'].apply(
            lambda x: 'Bay Area' if x > 37.3 else 'SoCal' if x < 34.5 else 'Central CA'
        )
        df.to_csv('data/housing_data.csv', index=False)
        print(f"  Loaded California Housing dataset: {len(df):,} real records")
        return df
    except ImportError:
        print("  sklearn not available, using synthetic data")
    except Exception as e:
        print(f"  Could not fetch California Housing ({type(e).__name__}), using synthetic")

    return _generate_synthetic(5000)

def _generate_synthetic(n=2000):
    """Generate realistic US housing market data with real neighborhood names."""
    rng = np.random.default_rng(42)

    neighborhoods = [
        'Beverly Hills', 'Palo Alto', 'Austin Heights', 'Denver Park', 'Phoenix Valley',
        'Atlanta Central', 'Chicago North', 'Seattle Eastside', 'Boston Back Bay', 'Miami Beach',
        'San Diego Coastal', 'Portland Pearl', 'Nashville Hills', 'Dallas Park Cities', 'Austin Downtown'
    ]

    # Realistic neighborhood quality tiers with price ranges
    neighborhood_list = rng.choice(neighborhoods, n)

    # Base price by neighborhood quality
    price_base = np.zeros(n)
    for i, nbr in enumerate(neighborhoods):
        mask = neighborhood_list == nbr
        if nbr in ['Beverly Hills', 'Palo Alto', 'Austin Heights']:
            # Luxury: $800K-$2M
            price_base[mask] = rng.uniform(800000, 2000000, mask.sum())
        elif nbr in ['Denver Park', 'Phoenix Valley', 'Atlanta Central']:
            # High-End: $500K-$900K
            price_base[mask] = rng.uniform(500000, 900000, mask.sum())
        elif nbr in ['Chicago North', 'Seattle Eastside', 'Boston Back Bay']:
            # Mid-Range: $300K-$600K
            price_base[mask] = rng.uniform(300000, 600000, mask.sum())
        elif nbr in ['Miami Beach', 'San Diego Coastal', 'Portland Pearl']:
            # Affordable: $150K-$350K
            price_base[mask] = rng.uniform(150000, 350000, mask.sum())
        else:
            # Economy: $80K-$200K
            price_base[mask] = rng.uniform(80000, 200000, mask.sum())

    bedrooms = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
    bathrooms = rng.choice([1, 1.5, 2, 2.5, 3, 4], n, p=[0.08, 0.12, 0.35, 0.20, 0.20, 0.05])
    sq_ft = (1500 + bedrooms * 600 + rng.normal(0, 800, n)).clip(600, 5000).astype(int)

    # Age (newer homes have higher prices)
    house_age = rng.integers(1, 81, n)
    age_factor = np.maximum(1 - house_age * 0.005, 0.4)  # Depreciation: ~0.5% per year

    has_garage = rng.choice([0, 1], n, p=[0.15, 0.85])
    has_pool = rng.choice([0, 1], n, p=[0.75, 0.25])

    price = price_base * (1 + (bedrooms - 3) * 0.08) * age_factor
    price = price * (1 + has_garage * 0.05) * (1 + has_pool * 0.10)
    price = price * (1 + sq_ft / 1500 * 0.05)
    price = price + rng.normal(0, 50000, n)
    price = np.clip(price, 80000, 2500000).round(0)

    # Derived features (income/value ratio proxy)
    median_income = (price / 35000).clip(0.5, 15.0)

    # Market density proxy
    population = rng.integers(500, 5000, n)
    avg_occupancy = (4 + rng.normal(0, 1.5, n)).clip(1.0, 8.0).round(1)

    df = pd.DataFrame({
        'neighborhood': neighborhood_list,
        'avg_bedrooms': bedrooms,
        'avg_bathrooms': bathrooms,
        'avg_rooms': sq_ft / 150,  # Approximate rooms from sq_ft
        'house_age': house_age,
        'has_garage': has_garage,
        'has_pool': has_pool,
        'median_income': median_income.round(2),
        'population': population,
        'avg_occupancy': avg_occupancy,
        'latitude': rng.uniform(25.0, 48.0, n).round(4),
        'longitude': rng.uniform(-125.0, -70.0, n).round(4),
        'price': price.astype(int),
    })

    # Add region based on neighborhood for SQL queries
    region_map = {
        'Beverly Hills': 'California', 'Palo Alto': 'California', 'Austin Heights': 'Texas',
        'Denver Park': 'Colorado', 'Phoenix Valley': 'Arizona', 'Atlanta Central': 'Georgia',
        'Chicago North': 'Illinois', 'Seattle Eastside': 'Washington', 'Boston Back Bay': 'Massachusetts',
        'Miami Beach': 'Florida', 'San Diego Coastal': 'California', 'Portland Pearl': 'Oregon',
        'Nashville Hills': 'Tennessee', 'Dallas Park Cities': 'Texas', 'Austin Downtown': 'Texas'
    }
    df['region'] = df['neighborhood'].map(region_map)

    df.to_csv('data/housing_data.csv', index=False)
    print(f"  Generated {n:,} realistic housing records")
    return df
