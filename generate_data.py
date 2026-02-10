"""
E-Commerce Recommendation System - Data Generator
Generates realistic synthetic e-commerce data for prototype
Author: Himanshu (XW013-25)
Course: Information Systems - WAI Project
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS = 1000
NUM_PRODUCTS = 200
NUM_RATINGS = 10000

# Product categories and names
CATEGORIES = [
    'Electronics', 'Fashion', 'Home & Kitchen', 'Books', 
    'Sports', 'Beauty', 'Toys', 'Grocery'
]

PRODUCT_PREFIXES = {
    'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smart Watch', 'Tablet', 'Camera', 'Gaming Console'],
    'Fashion': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket', 'Handbag', 'Sunglasses'],
    'Home & Kitchen': ['Mixer Grinder', 'Pressure Cooker', 'Bed Sheet', 'Curtains', 'Dinner Set', 'Vacuum Cleaner'],
    'Books': ['Novel', 'Self-Help Book', 'Cookbook', 'Biography', 'Business Book', 'Fiction'],
    'Sports': ['Running Shoes', 'Yoga Mat', 'Dumbbell Set', 'Cricket Bat', 'Football', 'Badminton Racket'],
    'Beauty': ['Face Cream', 'Shampoo', 'Lipstick', 'Perfume', 'Skincare Kit', 'Hair Oil'],
    'Toys': ['Action Figure', 'Board Game', 'Puzzle', 'Remote Car', 'Doll', 'Building Blocks'],
    'Grocery': ['Rice', 'Cooking Oil', 'Tea', 'Coffee', 'Snacks', 'Spices']
}

BRANDS = ['Samsung', 'Apple', 'Sony', 'LG', 'Nike', 'Adidas', 'Puma', 'HP', 'Dell', 'Generic']

print("="*60)
print("E-COMMERCE RECOMMENDATION SYSTEM")
print("Data Generator - Creating Synthetic Dataset")
print("="*60)
print()

# Generate Products
print("Step 1: Generating Products...")
products_data = []
for i in range(NUM_PRODUCTS):
    category = random.choice(CATEGORIES)
    prefix = random.choice(PRODUCT_PREFIXES[category])
    brand = random.choice(BRANDS)
    
    product = {
        'product_id': f'P{i+1:04d}',
        'product_name': f'{brand} {prefix}',
        'category': category,
        'price': round(random.uniform(500, 50000), 2),
        'brand': brand
    }
    products_data.append(product)

products_df = pd.DataFrame(products_data)
print(f"✓ Generated {len(products_df)} products across {len(CATEGORIES)} categories")

# Generate Users
print("\nStep 2: Generating Users...")
users_data = []
for i in range(NUM_USERS):
    user = {
        'user_id': f'U{i+1:05d}',
        'user_name': f'Customer_{i+1}',
        'age_group': random.choice(['18-25', '26-35', '36-45', '46-60', '60+']),
        'location': random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Hyderabad', 'Chennai'])
    }
    users_data.append(user)

users_df = pd.DataFrame(users_data)
print(f"✓ Generated {len(users_df)} users from 6 major cities")

# Generate Ratings with realistic patterns
print("\nStep 3: Generating Ratings (with realistic user preferences)...")

# Each user has preferred categories
user_category_preferences = {}
for user_id in users_df['user_id']:
    num_prefs = random.randint(1, 3)
    user_category_preferences[user_id] = random.sample(CATEGORIES, num_prefs)

ratings_data = []
existing_pairs = set()

while len(ratings_data) < NUM_RATINGS:
    user_id = random.choice(users_df['user_id'].tolist())
    
    # 70% chance user rates products from preferred categories
    if random.random() < 0.7 and user_id in user_category_preferences:
        preferred_cats = user_category_preferences[user_id]
        product = products_df[products_df['category'].isin(preferred_cats)].sample(1).iloc[0]
        # Higher ratings for preferred categories
        rating = random.choices([3, 4, 5], weights=[0.2, 0.3, 0.5])[0]
    else:
        product = products_df.sample(1).iloc[0]
        # Normal distribution of ratings
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0]
    
    pair = (user_id, product['product_id'])
    
    # Avoid duplicate user-product pairs
    if pair not in existing_pairs:
        existing_pairs.add(pair)
        
        # Generate timestamp (last 6 months)
        days_ago = random.randint(0, 180)
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        rating_record = {
            'user_id': user_id,
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'category': product['category'],
            'rating': rating,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        ratings_data.append(rating_record)

ratings_df = pd.DataFrame(ratings_data)
print(f"✓ Generated {len(ratings_df)} ratings with realistic patterns")

# Save to CSV files
print("\nStep 4: Saving data to CSV files...")
products_df.to_csv('products.csv', index=False)
users_df.to_csv('users.csv', index=False)
ratings_df.to_csv('ratings.csv', index=False)
print("✓ Saved: products.csv")
print("✓ Saved: users.csv")
print("✓ Saved: ratings.csv")

# Display statistics
print("\n" + "="*60)
print("DATASET GENERATION COMPLETE!")
print("="*60)
print("\n📊 Dataset Statistics:")
print(f"  • Total Users: {len(users_df):,}")
print(f"  • Total Products: {len(products_df):,}")
print(f"  • Total Ratings: {len(ratings_df):,}")
print(f"  • Average Rating: {ratings_df['rating'].mean():.2f}/5.0")
print(f"  • Matrix Sparsity: {((len(users_df)*len(products_df)-len(ratings_df))/(len(users_df)*len(products_df)))*100:.1f}%")

print("\n⭐ Rating Distribution:")
for rating in [1, 2, 3, 4, 5]:
    count = len(ratings_df[ratings_df['rating'] == rating])
    percentage = (count / len(ratings_df)) * 100
    bar = '█' * int(percentage / 2)
    print(f"  {rating} stars: {bar} {count:4d} ({percentage:5.1f}%)")

print("\n📦 Category Distribution:")
cat_dist = products_df['category'].value_counts()
for cat, count in cat_dist.items():
    print(f"  {cat:20s}: {count:3d} products")

print("\n✅ Ready for Recommendation Engine!")
print("   Next step: Run 'python3 recommendation_engine.py'")
print()
