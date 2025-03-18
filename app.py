import streamlit as st
from backend.models.main import Recommender  # Import Recommender from main.py

def run_streamlit_app():
    st.title("BestBuy: AI-Powered Price Comparison and Product Recommendation System")
    st.write("Find the best deals on quality products within your budget!")

    query = st.text_input("Enter product (e.g., 'affordable phone')", value="affordable phone fit for content creation")
    budget = st.number_input("Enter your budget (₦)", min_value=1000, max_value=2_000_000, value=105_000, step=1000)
    num_results = st.slider("Number of results", min_value=1, max_value=5, value=2)
    source_filter = st.selectbox("Filter by source", ["All", "Jumia", "Konga"], index=0)

    if st.button("Get Recommendations"):
        # Input validation
        if not query.strip():
            st.error("Please enter a product query.")
        elif budget < 1000:
            st.error("Budget must be at least ₦1000.")
        else:
            st.write("### Recommendations")
            with st.spinner("Fetching recommendations..."):
                recommender = Recommender()
                recommendation = recommender.get_recommendations(query, budget, num_results, source_filter)
                st.markdown(recommendation, unsafe_allow_html=True)

if __name__ == "__main__":
    run_streamlit_app()