import os
import csv
import time
from dotenv import load_dotenv
import sqlite3
import chromadb
from openai import OpenAI
import logging
from functools import lru_cache
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Paths
DATA_PATH = "C:/Code/neural_drip/data"
CSV_PATH = os.path.join(DATA_PATH, "jumia_and_konga_data2.csv")
VECTOR_DB_PATH = "C:/Code/neural_drip/backend/vector_db"

# Initialize OpenAI and ChromaDB clients
client = OpenAI(api_key=openai_api_key, timeout=30)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

class DataEmbedder:
    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = csv_path
        self.products = []
        self.collection = None

    def load_and_process_data(self):
        logger.info(f"Attempting to open CSV at: {self.csv_path}")
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                logger.info(f"CSV headers: {reader.fieldnames}")
                
                for i, row in enumerate(reader):
                    try:
                        price_str = row["price"].replace("₦", "").replace("NGN", "").replace(",", "").strip()
                        price = float(price_str.split("-")[0].strip()) if "-" in price_str else float(price_str)

                        rating_str = row.get("ratings", "").strip()
                        rating = (float(rating_str.split(" out of ")[0]) if "out of" in rating_str else
                                  float(rating_str) if rating_str.replace(".", "").isdigit() else 0.0)

                        discount_str = row.get("discount", "0").strip()
                        discount = (0.0 if discount_str == "No Discount" else
                                    float(discount_str.replace("%", "")) / 100 if "%" in discount_str else
                                    float(discount_str) / 100 if discount_str.isdigit() else 0.0)

                        if price < 0 or discount > 1:
                            safe_name = row['product_name'].encode('ascii', 'ignore').decode('ascii')
                            logger.warning(f"Invalid data for {safe_name}: price={price}, discount={discount}")
                            continue

                        source = row.get("source", "Unknown").capitalize()
                        self.products.append({
                            "product_name": row["product_name"],
                            "price_ngn": price,
                            "discount": discount,
                            "rating": rating,
                            "product_link": row["product_link"],
                            "source": source
                        })
                        safe_name = row['product_name'].encode('ascii', 'ignore').decode('ascii')
                        logger.info(f"Successfully processed: {safe_name}")

                    except (ValueError, KeyError) as e:
                        safe_name = row.get('product_name', 'Unknown').encode('ascii', 'ignore').decode('ascii')
                        logger.error(f"Error processing {safe_name}: {str(e)}")
                        continue

        except FileNotFoundError:
            logger.error(f"CSV file not found at {self.csv_path}")
            raise

        if not self.products:
            logger.error(f"No valid products processed from {self.csv_path}")
            raise ValueError(f"No valid products processed from {self.csv_path}.")

    def embed_data(self):
        texts = [f"{p['product_name']} - ₦{p['price_ngn']} - Discount: {p['discount']*100}% - Rating: {p['rating']} - Source: {p['source']}" 
                 for p in self.products]
        ids = [f"prod_{i}" for i in range(len(self.products))]
        metadatas = [{"price_ngn": p["price_ngn"], "discount": p["discount"], "rating": p["rating"], 
                      "url": p["product_link"], "source": p["source"]} for p in self.products]

        invalid_texts = [t for t in texts if not isinstance(t, str) or not t.strip()]
        if invalid_texts:
            logger.error(f"Found {len(invalid_texts)} invalid texts: {invalid_texts[:5]}")
            raise ValueError(f"Found {len(invalid_texts)} invalid texts: {invalid_texts[:5]}")

        batch_size = 166
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1} of {len(texts) // batch_size + 1} ({len(batch)} items)")
            response = client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings.extend([r.embedding for r in response.data])

        try:
            chroma_client.delete_collection("products")
        except Exception as e:
            logger.warning(f"Couldn't delete existing collection: {e}")

        self.collection = chroma_client.create_collection(name="products")
        for i in range(0, len(embeddings), batch_size):
            batch_embeds = embeddings[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            self.collection.add(embeddings=batch_embeds, documents=batch_texts, metadatas=batch_metas, ids=batch_ids)

        logger.info(f"Data embedded and stored! {len(self.products)} products processed.")

class Recommender:
    def __init__(self):
        self.collection = chroma_client.get_collection(name="products")

    def prioritize_high_specs(self, products):
        return sorted(products, key=lambda x: (x["metadata"].get("rating", 0), x["metadata"].get("price_ngn", 0)), reverse=True)

    @lru_cache(maxsize=128)
    def get_query_embedding(self, query):
        """Cached function to generate query embedding."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(5)
                else:
                    logger.error(f"Failed to embed query after {max_retries} attempts: {str(e)}")
                    raise

    def calculate_diversity_score(self, products, selected_results):
        """Score products based on brand and price variety."""
        if not products:
            return []

        selected_brands = Counter(r["document"].split(" - ")[0].split()[0] for r in selected_results)
        selected_prices = [r["metadata"]["price_ngn"] for r in selected_results]

        scored_products = []
        for product in products:
            brand = product["document"].split(" - ")[0].split()[0]
            price = product["metadata"]["price_ngn"]
            
            brand_diversity = 1 / (selected_brands[brand] + 1) if brand in selected_brands else 1
            if selected_prices:
                avg_price = sum(selected_prices) / len(selected_prices)
                price_diversity = min(1, abs(price - avg_price) / 10000)
            else:
                price_diversity = 1
            
            score = brand_diversity * 0.6 + price_diversity * 0.4
            scored_products.append((score, product))

        return [p for _, p in sorted(scored_products, key=lambda x: x[0], reverse=True)]

    @lru_cache(maxsize=128)
    def get_recommendations(self, query, budget, num_results, source_filter="All"):
        max_retries = 3
        query_embedding = self.get_query_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=100)

        filtered_results = []
        for id, meta, doc in zip(results["ids"][0], results["metadatas"][0], results["documents"][0]):
            price = meta["price_ngn"]
            discount = meta.get("discount", 0)
            effective_price = price * (1 - discount / 100)
            if (effective_price <= budget * 1.1) and (source_filter == "All" or meta["source"] == source_filter.capitalize()):
                filtered_results.append({"id": id, "metadata": meta, "document": doc})

        if not filtered_results:
            return f"No products found within your budget of ₦{budget} matching your query '{query}' from {source_filter}."

        if "content creator" in query.lower():
            filtered_results = self.prioritize_high_specs(filtered_results)

        selected_results = []
        remaining_results = filtered_results.copy()
        while len(selected_results) < num_results and remaining_results:
            remaining_results = self.calculate_diversity_score(remaining_results, selected_results)
            if remaining_results:
                selected_results.append(remaining_results.pop(0))

        if not selected_results:
            return f"No products found within your budget of ₦{budget} matching your query '{query}' from {source_filter}."

        context = "\n".join([f"- Product: {r['document']}, Rating: {r['metadata']['rating']}, Price: ₦{r['metadata']['price_ngn']}, "
                             f"Discount: {r['metadata'].get('discount', 0)}%, Source: {r['metadata']['source']}, Link: {r['metadata']['url']}"
                             for r in selected_results])

        prompt = (f"You are an expert shopping assistant. Given this product data:\n{context}\n"
                  f"Recommend the best options for '{query}' within a budget of ₦{budget}. "
                  f"If the query mentions 'content creator', prioritize high specs like camera quality, storage, and performance. "
                  f"Consider price, discount, rating, and source (Jumia or Konga). Provide exactly {num_results} diverse recommendations if available, "
                  f"each with a clear explanation of why it’s a good choice. Include alternatives if applicable. "
                  f"Format your response as a numbered list with this structure for each item:\n"
                  f"X. **Product Name** - Price: ₦XXX - Discount: XX% - Rating: X.X - Source: [Jumia/Konga]\n"
                  f"   - Reason: [Why it’s recommended]\n"
                  f"   - Link: <a href='[URL]'>Buy Now</a>\n"
                  f"Do not recommend products exceeding the budget of ₦{budget}, but include slightly above-budget items if they have significant discounts.")

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(5)
                else:
                    logger.error(f"Failed to get LLM response after {max_retries} attempts: {str(e)}")
                    raise

if __name__ == "__main__":
    # Uncomment to embed data once
    embedder = DataEmbedder()
    embedder.load_and_process_data()
    embedder.embed_data()