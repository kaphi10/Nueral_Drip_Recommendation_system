# AI-Powered Price Comparison & Product Recommendation System

**Overview**
Welcome to the AI-Powered Price Comparison & Product Recommendation System! This project aims to help Nigerians shop smarter by comparing product prices across platforms like Jumia, Konga, Jiji, and Slot in real-time. Our AI analyzes price, durability, and user reviews to recommend the best value-for-money options, empowering users to save money without compromising quality.

**Problem Statement**
With nearly 50% of Nigerians living below the average income (World Bank), finding affordable, durable products is a challenge. This system addresses:
1. Wide price variations across platforms.
2. Time-consuming manual searches.
3. Low-quality, wasteful purchases.

**Core Features**
 1. Real-Time Price Comparison: Scrapes prices from multiple Nigerian e-commerce platforms.
 2. Best Value Recommendations: AI ranks products by price, durability, and reviews.
 3. Budget-Based Search: Filters products within user-defined budgets.
 4. Alternative Suggestions: Offers cheaper or higher-quality options.
 5. Customizable Results: Users choose how many top products to view (e.g., Top 5) with direct purchase links.

 **Project Structure**
 SmartPickNG/
├── scrapers/           # Web scraping scripts (e.g., jumia.py, konga.py)
├── ai/                # AI logic (e.g., recommender.py, sentiment.py)
├── frontend/          # Flask app and UI (e.g., app.py, templates/, static/)
├── tests/             # Test scripts (e.g., test_scrapers.py)
├── README.md          # This file
└── .gitignore         # Ignored files (e.g., __pycache__, venv/) 







