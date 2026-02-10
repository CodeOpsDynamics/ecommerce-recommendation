"""
E-Commerce Recommendation Engine - Core Algorithm
Implements Collaborative Filtering using K-Nearest Neighbors
Author: Himanshu (XW013-25)
Course: Information Systems - WAI Project
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    """
    Product Recommendation Engine using Collaborative Filtering
    
    Algorithm: K-Nearest Neighbors (KNN) with Cosine Similarity
    Approach: User-based Collaborative Filtering
    """
    
    def __init__(self, ratings_df, products_df):
        """Initialize the recommendation engine"""
        self.ratings_df = ratings_df
        self.products_df = products_df
        self.user_item_matrix = None
        self.model = None
        
        print("🤖 Initializing Recommendation Engine...")
        self._build_user_item_matrix()
        self._train_model()
        print("✅ Recommendation Engine Ready!\n")
    
    def _build_user_item_matrix(self):
        """Create user-item matrix (users as rows, products as columns)"""
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )
        
        print(f"  📊 User-Item Matrix: {self.user_item_matrix.shape[0]} users × {self.user_item_matrix.shape[1]} products")
        print(f"  📉 Sparsity: {self._calculate_sparsity():.2f}%")
    
    def _calculate_sparsity(self):
        """Calculate matrix sparsity (percentage of zeros)"""
        total = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        non_zero = np.count_nonzero(self.user_item_matrix.values)
        return ((total - non_zero) / total) * 100
    
    def _train_model(self):
        """Train KNN model for finding similar users"""
        self.model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=10
        )
        self.model.fit(self.user_item_matrix.values)
        print(f"  🎯 KNN Model Trained (K=10, metric=cosine)")
    
    def get_user_recommendations(self, user_id, n_recommendations=5):
        """Get product recommendations for a specific user"""
        if user_id not in self.user_item_matrix.index:
            print(f"  ⚠️  User {user_id} not found, showing popular products")
            return self._get_popular_products(n_recommendations)
        
        # Get user's rating vector
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # Find similar users
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=11)
        similar_user_indices = indices[0][1:]  # Exclude user itself
        
        # Generate recommendations
        recommendations = self._aggregate_similar_user_ratings(
            user_idx,
            similar_user_indices,
            n_recommendations
        )
        
        return recommendations
    
    def _aggregate_similar_user_ratings(self, user_idx, similar_users, n_recs):
        """Aggregate ratings from similar users"""
        user_rated_products = set(
            self.user_item_matrix.columns[self.user_item_matrix.iloc[user_idx] > 0]
        )
        
        product_scores = {}
        
        for similar_user_idx in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]
            highly_rated = similar_user_ratings[similar_user_ratings >= 4]
            
            for product_id, rating in highly_rated.items():
                if product_id not in user_rated_products:
                    if product_id not in product_scores:
                        product_scores[product_id] = []
                    product_scores[product_id].append(rating)
        
        # Calculate average scores
        avg_scores = {
            product: np.mean(scores) 
            for product, scores in product_scores.items()
        }
        
        # Get top N
        top_products = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]
        
        # Create DataFrame
        recommended_ids = [p[0] for p in top_products]
        predicted_ratings = [p[1] for p in top_products]
        
        recommendations = self.products_df[
            self.products_df['product_id'].isin(recommended_ids)
        ].copy()
        
        recommendations['predicted_rating'] = recommendations['product_id'].map(
            dict(zip(recommended_ids, predicted_ratings))
        )
        
        return recommendations.sort_values('predicted_rating', ascending=False)
    
    def _get_popular_products(self, n_recommendations):
        """Fallback: Return most popular products"""
        popular = self.ratings_df.groupby('product_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        popular.columns = ['product_id', 'avg_rating', 'rating_count']
        popular = popular[popular['rating_count'] >= 10]
        popular = popular.sort_values('avg_rating', ascending=False)
        
        top_products = popular.head(n_recommendations)
        recommendations = self.products_df[
            self.products_df['product_id'].isin(top_products['product_id'])
        ].merge(top_products, on='product_id')
        
        recommendations['predicted_rating'] = recommendations['avg_rating']
        return recommendations
    
    def get_model_statistics(self):
        """Get model statistics"""
        return {
            'total_users': len(self.user_item_matrix),
            'total_products': len(self.user_item_matrix.columns),
            'total_ratings': len(self.ratings_df),
            'sparsity': self._calculate_sparsity(),
            'avg_rating': self.ratings_df['rating'].mean(),
            'avg_ratings_per_user': len(self.ratings_df) / len(self.user_item_matrix)
        }


# Test the recommendation engine
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING RECOMMENDATION ENGINE")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading data...")
    try:
        ratings_df = pd.read_csv('ratings.csv')
        products_df = pd.read_csv('products.csv')
        users_df = pd.read_csv('users.csv')
        print(f"  ✓ Loaded {len(ratings_df):,} ratings")
        print(f"  ✓ Loaded {len(products_df):,} products")
        print(f"  ✓ Loaded {len(users_df):,} users\n")
    except FileNotFoundError:
        print("❌ Error: Data files not found!")
        print("   Please run 'python3 generate_data.py' first.\n")
        exit(1)
    
    # Initialize engine
    engine = RecommendationEngine(ratings_df, products_df)
    
    # Get recommendations for sample users
    print("="*70)
    print("GENERATING SAMPLE RECOMMENDATIONS")
    print("="*70 + "\n")
    
    for i in range(3):
        sample_user = users_df.iloc[i]['user_id']
        user_location = users_df.iloc[i]['location']
        
        print(f"👤 User: {sample_user} (Location: {user_location})")
        print("-" * 70)
        
        # Show user's purchase history
        user_history = ratings_df[ratings_df['user_id'] == sample_user].sort_values('rating', ascending=False)
        print("📜 Purchase History (Top 3):")
        for idx, row in user_history.head(3).iterrows():
            print(f"  • {row['product_name']} ({row['category']}) - {'⭐' * row['rating']}")
        
        print("\n🎁 AI Recommendations:")
        recommendations = engine.get_user_recommendations(sample_user, n_recommendations=5)
        
        for idx, row in recommendations.head(5).iterrows():
            stars = '⭐' * int(row['predicted_rating'])
            print(f"  {idx+1}. {row['product_name']}")
            print(f"     Category: {row['category']} | Price: ₹{row['price']:,.0f} | Predicted: {stars} ({row['predicted_rating']:.2f})")
        
        print("\n" + "="*70 + "\n")
    
    # Model statistics
    print("📊 MODEL STATISTICS")
    print("="*70)
    stats = engine.get_model_statistics()
    print(f"  • Total Users: {stats['total_users']:,}")
    print(f"  • Total Products: {stats['total_products']:,}")
    print(f"  • Total Ratings: {stats['total_ratings']:,}")
    print(f"  • Matrix Sparsity: {stats['sparsity']:.2f}%")
    print(f"  • Average Rating: {stats['avg_rating']:.2f}/5.0")
    print(f"  • Avg Ratings/User: {stats['avg_ratings_per_user']:.1f}")
    
    print("\n✅ Recommendation Engine Test Complete!")
    print("   Next step: Run 'streamlit run app.py' to see the web interface\n")
