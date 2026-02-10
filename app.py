"""
E-Commerce Recommendation System - Interactive Web App
Streamlit Application
Author: Himanshu (XW013-25)
Course: Information Systems - WAI Project
Prof. Anupriya Khan
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommendation_engine import RecommendationEngine
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Product Recommendations",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 24px;
        color: #4ECDC4;
        font-weight: bold;
        margin-top: 20px;
    }
    .product-card {
        border: 2px solid #4ECDC4;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load data and initialize engine
@st.cache_resource
def load_system():
    """Load data and initialize recommendation engine (cached for performance)"""
    try:
        ratings_df = pd.read_csv('ratings.csv')
        products_df = pd.read_csv('products.csv')
        users_df = pd.read_csv('users.csv')
        engine = RecommendationEngine(ratings_df, products_df)
        return engine, ratings_df, products_df, users_df, True
    except FileNotFoundError:
        return None, None, None, None, False

# Load system
engine, ratings_df, products_df, users_df, data_loaded = load_system()

def main():
    # Header
    st.markdown('<div class="main-header">🛍️ AI-Powered Product Recommendation System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 10px; font-size: 16px;'>
    <b>Intelligent E-Commerce Recommendations using Collaborative Filtering & Machine Learning</b>
    <br><i>Information Systems - Working with AI Project | IIM Ranchi | Himanshu (XW013-25)</i>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if data loaded
    if not data_loaded:
        st.error("⚠️ **Data files not found!**")
        st.info("📌 Please run `python3 generate_data.py` first to create the dataset.")
        st.code("python3 generate_data.py", language="bash")
        return
    
    # Sidebar navigation
    st.sidebar.markdown("### 🎯 Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "🎯 Get Recommendations", "📊 Analytics", "⚙️ How It Works"],
        label_visibility="collapsed"
    )
    
    # Route to pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Get Recommendations":
        show_recommendations_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ How It Works":
        show_how_it_works_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Project Info")
    st.sidebar.info("""
    **Student:** Himanshu  
    **ID:** XW013-25  
    **Course:** Information Systems  
    **Professor:** Prof. Anupriya Khan  
    **Institution:** IIM Ranchi
    """)

def show_home_page():
    """Display home page with overview"""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    stats = engine.get_model_statistics()
    
    with col1:
        st.metric("👥 Total Users", f"{int(stats['total_users']):,}")
    with col2:
        st.metric("📦 Total Products", f"{int(stats['total_products']):,}")
    with col3:
        st.metric("⭐ Total Ratings", f"{int(stats['total_ratings']):,}")
    with col4:
        st.metric("📈 Avg Rating", f"{stats['avg_rating']:.2f}/5.0")
    
    st.markdown("---")
    
    # Problem & Solution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ❌ Problem: Without AI")
        st.markdown("""
        **Traditional E-Commerce Experience:**
        - 🔍 Customers browse randomly through 1000s of products
        - ⏰ Average search time: **15-20 minutes**
        - 📉 Low conversion rate: **2-3%**
        - 😓 Poor customer satisfaction
        - 💸 Lost revenue opportunities
        - 🛒 High cart abandonment: **70%**
        """)
    
    with col2:
        st.markdown("### ✅ Solution: With AI")
        st.markdown("""
        **AI-Powered Personalized Experience:**
        - 🎯 Smart product recommendations based on behavior
        - ⚡ Reduced search time: **3-5 minutes** (-70%)
        - 📈 Increased conversion: **8-12%** (+300%)
        - 😊 Enhanced customer satisfaction
        - 💰 Higher revenue per customer
        - ✨ Better shopping experience
        """)
    
    st.markdown("---")
    
    # Business Impact
    st.markdown("### 💼 Business Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Revenue Growth**
        - 15-25% revenue increase
        - Higher average order value
        - More repeat purchases
        """)
    
    with col2:
        st.markdown("""
        **⏱️ Efficiency Gains**
        - 70% reduction in browse time
        - Better product discovery
        - Faster purchase decisions
        """)
    
    with col3:
        st.markdown("""
        **😊 Customer Satisfaction**
        - 30% improvement in retention
        - Personalized experience
        - Reduced frustration
        """)
    
    st.markdown("---")
    
    # Technology
    st.markdown("### 🤖 Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Algorithm:**
        - K-Nearest Neighbors (KNN)
        - Collaborative Filtering
        - Cosine Similarity
        - User-based recommendations
        """)
    
    with col2:
        st.markdown("""
        **Implementation:**
        - Python + Scikit-learn
        - Pandas + NumPy
        - Streamlit (Web UI)
        - Plotly (Visualizations)
        """)

def show_recommendations_page():
    """Generate and display recommendations"""
    
    st.markdown("### 🎯 Get Personalized Product Recommendations")
    
    # User selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_list = users_df['user_id'].tolist()
        selected_user = st.selectbox(
            "Select a User:",
            user_list,
            format_func=lambda x: f"{x} - {users_df[users_df['user_id']==x]['location'].values[0]}"
        )
    
    with col2:
        num_recs = st.slider("Number of Recommendations", 3, 10, 5)
    
    if st.button("🎁 Generate Recommendations", type="primary", use_container_width=True):
        
        with st.spinner("🔍 Analyzing user preferences..."):
            
            # User info
            user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
            st.markdown(f"#### 👤 User Profile: {selected_user}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Location", user_info['location'])
            col2.metric("Age Group", user_info['age_group'])
            
            # Purchase history
            st.markdown("---")
            st.markdown("### 📜 Purchase History")
            
            user_history = ratings_df[ratings_df['user_id'] == selected_user]
            
            if len(user_history) > 0:
                history_display = user_history.merge(
                    products_df[['product_id', 'product_name', 'category', 'price']], 
                    on='product_id'
                )
                history_display = history_display.sort_values('rating', ascending=False)
                
                for idx, row in history_display.head(5).iterrows():
                    col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
                    with col1:
                        st.write(f"**{row['product_name_y']}**")
                    with col2:
                        st.write(f"📦 {row['category_y']}")
                    with col3:
                        st.write(f"₹{row['price']:,.0f}")
                    with col4:
                        st.write("⭐" * int(row['rating']))
            else:
                st.info("No purchase history available for this user.")
            
            # Generate recommendations
            st.markdown("---")
            st.markdown("### 🎁 AI-Generated Recommendations")
            
            recommendations = engine.get_user_recommendations(selected_user, n_recommendations=num_recs)
            
            # Display as cards
            for idx, row in recommendations.iterrows():
                st.markdown(f"""
                <div class="product-card">
                    <h3>🛍️ {row['product_name']}</h3>
                    <p style='font-size: 16px;'>
                        <b>Category:</b> {row['category']} | 
                        <b>Brand:</b> {row['brand']} | 
                        <b>Price:</b> ₹{row['price']:,.2f}
                    </p>
                    <p style='font-size: 18px; color: #FF6B6B;'>
                        <b>Predicted Rating:</b> {'⭐' * int(row['predicted_rating'])} 
                        ({row['predicted_rating']:.2f}/5.0)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success(f"✅ Successfully generated {len(recommendations)} personalized recommendations!")

def show_analytics_page():
    """Display analytics and insights"""
    
    st.markdown("### 📊 Analytics Dashboard")
    
    # Rating distribution
    st.markdown("#### ⭐ Rating Distribution")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    
    fig_ratings = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating (Stars)', 'y': 'Number of Ratings'},
        title='How Users Rate Products',
        color=rating_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_ratings.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_ratings, use_container_width=True)
    
    # Category analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📦 Products by Category")
        category_counts = products_df['category'].value_counts()
        fig_cat = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Product Distribution'
        )
        fig_cat.update_layout(height=400)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.markdown("#### 🛍️ Ratings by Category")
        ratings_by_cat = ratings_df['category'].value_counts()
        fig_rat_cat = px.pie(
            values=ratings_by_cat.values,
            names=ratings_by_cat.index,
            title='Most Rated Categories'
        )
        fig_rat_cat.update_layout(height=400)
        st.plotly_chart(fig_rat_cat, use_container_width=True)
    
    # Top products
    st.markdown("---")
    st.markdown("#### 🏆 Top Rated Products")
    
    top_products = ratings_df.groupby('product_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    top_products.columns = ['product_id', 'avg_rating', 'num_ratings']
    top_products = top_products[top_products['num_ratings'] >= 5]
    top_products = top_products.sort_values('avg_rating', ascending=False).head(10)
    
    top_display = top_products.merge(products_df, on='product_id')
    st.dataframe(
        top_display[['product_name', 'category', 'brand', 'price', 'avg_rating', 'num_ratings']]
        .style.format({'price': '₹{:,.2f}', 'avg_rating': '{:.2f}'}),
        use_container_width=True,
        height=400
    )

def show_how_it_works_page():
    """Explain the algorithm"""
    
    st.markdown("### ⚙️ How the AI Recommendation System Works")
    
    st.markdown("""
    ## 🤖 Collaborative Filtering with K-Nearest Neighbors
    
    Our system uses **User-Based Collaborative Filtering** to generate personalized product recommendations.
    
    ### 📋 Step-by-Step Process:
    """)
    
    # Step 1
    st.markdown("""
    #### 1️⃣ Build User-Item Matrix
    We create a matrix where:
    - **Rows** = Users (1000 users)
    - **Columns** = Products (200 products)
    - **Values** = Ratings (1-5 stars)
    - Missing ratings are filled with 0
    """)
    
    # Step 2
    st.markdown("""
    #### 2️⃣ Find Similar Users
    Using **Cosine Similarity**, we find users with similar tastes:
    
    ```
    Similarity = (User A · User B) / (||User A|| × ||User B||)
    ```
    
    - Values range from 0 (completely different) to 1 (identical)
    - We find the K=10 most similar users using KNN
    """)
    
    # Step 3
    st.markdown("""
    #### 3️⃣ Generate Recommendations
    From similar users, we:
    - Look at products they rated highly (4-5 stars)
    - Exclude products the target user already rated
    - Calculate predicted ratings as average of similar users' ratings
    - Rank products by predicted rating
    """)
    
    # Step 4
    st.markdown("""
    #### 4️⃣ Return Top N Products
    We return the top 5-10 products with:
    - Predicted ratings
    - Product details
    - Category and price information
    """)
    
    st.markdown("---")
    
    # Why this algorithm
    st.markdown("""
    ### 🎯 Why This Algorithm?
    
    **Advantages:**
    - ✅ Learns from real user behavior
    - ✅ Discovers hidden patterns
    - ✅ Works well with sparse data (95% sparsity!)
    - ✅ No need for product metadata
    - ✅ Adapts to changing preferences
    
    **Real-World Usage:**
    - 🛒 **Amazon:** "Customers who bought this also bought..."
    - 🎬 **Netflix:** "Because you watched..."
    - 🎵 **Spotify:** "Discover Weekly"
    - 📘 **Facebook:** "People you may know"
    """)
    
    st.markdown("---")
    
    # Model performance
    st.markdown("### 📈 Model Performance")
    
    stats = engine.get_model_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Matrix Sparsity", f"{stats['sparsity']:.1f}%")
        st.caption("Percentage of missing ratings")
    
    with col2:
        st.metric("Avg Ratings/User", f"{stats['avg_ratings_per_user']:.1f}")
        st.caption("User engagement level")
    
    with col3:
        st.metric("Data Coverage", f"{100-stats['sparsity']:.1f}%")
        st.caption("Available rating data")
    
    st.markdown("---")
    
    # Algorithm parameters
    st.markdown("""
    ### 🔧 Algorithm Configuration
    
    | Parameter | Value | Why? |
    |-----------|-------|------|
    | **K (Neighbors)** | 10 | Balance between accuracy and diversity |
    | **Similarity Metric** | Cosine | Handles rating scale differences well |
    | **Minimum Rating** | 4.0 | Only recommend highly-rated products |
    | **Algorithm** | Brute Force | Accuracy over speed for prototype |
    """)

# Run the app
if __name__ == "__main__":
    main()
