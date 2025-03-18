import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
import time
import random

# load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# initialize OpenAI client with timeout and retries
path="C:/Users/USER/Desktop/llm_exe/Neural-Drip/backend/vector_db"
client = OpenAI(api_key=openai_api_key, timeout=30)
chroma_client = chromadb.PersistentClient(path=path)
collection = chroma_client.get_collection(name="products")


def prioritize_high_specs(products):
    """Sorts products by high specs if the users query is related to it."""
    return sorted(
        products,
        key=lambda x: (x["metadata"].get("rating", 0), x["metadata"].get("price_ngn", 0)),
        reverse=True
    )


def get_recommendations(query, budget, num_results, source_filter="All"):
    """
    Get diverse product recommendations based on a query, budget, number of results, and source filter.
    
    Args:
        query (str): User search query (e.g., "iPhone with good specs for a content creator")
        budget (float): Maximum price in Naira
        num_results (int): Number of products to return
        source_filter (str): Filter by source ("All", "Jumia", or "Konga")
    
    Returns:
        str: LLM-generated recommendation text
    """
    # embed the query with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise Exception(f"Failed to embed query after {max_retries} attempts: {str(e)}")

    # query the vector DB, fetch more results for diversity
    results = collection.query(query_embeddings=[query_embedding], n_results=100)  # Increased to 100 for better selection

    # filter by budget and source with some flexibility (allowing products slightly above budget with high discounts)
    filtered_results = []
    for id, meta, doc in zip(results["ids"][0], results["metadatas"][0], results["documents"][0]):
        price = meta["price_ngn"]
        discount = meta.get("discount", 0)  # Handle cases where discount might be missing
        effective_price = price * (1 - discount / 100)  # Adjust price based on discount

        if (effective_price <= budget * 1.1) and (source_filter == "All" or meta["source"] == source_filter.capitalize()):
            filtered_results.append({"id": id, "metadata": meta, "document": doc})

    if not filtered_results:
        return f"No products found within your budget of ₦{budget} matching your query '{query}' from {source_filter}."

    # Prioritize high-spec devices if the query is for content creators
    if "content creator" in query.lower():
        filtered_results = prioritize_high_specs(filtered_results)

    random.shuffle(filtered_results)  # Introduce randomness for diversity

    # Ensure a balanced selection from Jumia and Konga
    if source_filter == "All":
        jumia_results = [r for r in filtered_results if r["metadata"]["source"] == "Jumia"]
        konga_results = [r for r in filtered_results if r["metadata"]["source"] == "Konga"]

        selected_results = []
        i, j = 0, 0
        while len(selected_results) < num_results and (i < len(jumia_results) or j < len(konga_results)):
            if i < len(jumia_results):
                selected_results.append(jumia_results[i])
                i += 1
            if j < len(konga_results) and len(selected_results) < num_results:
                selected_results.append(konga_results[j])
                j += 1
    else:
        selected_results = filtered_results[:num_results]

    if not selected_results:
        return f"No products found within your budget of ₦{budget} matching your query '{query}' from {source_filter}."

    # Prepare context for LLM
    context = "\n".join([
        f"- Product: {r['document']}, Rating: {r['metadata']['rating']}, Price: ₦{r['metadata']['price_ngn']}, "
        f"Discount: {r['metadata'].get('discount', 0)}%, Source: {r['metadata']['source']}, Link: {r['metadata']['url']}"
        for r in selected_results
    ])

    # Enhanced prompt for content creator needs and budget utilization
    prompt = (
        f"You are an expert shopping assistant. Given this product data:\n{context}\n"
        f"Recommend the best options for '{query}' within a budget of ₦{budget}. "
        f"If the query mentions 'content creator', prioritize high specs like camera quality, storage, and performance. "
        f"Consider price, discount, rating, and source (Jumia or Konga). Provide exactly {num_results} diverse recommendations if available, "
        f"each with a clear explanation of why it’s a good choice. Include alternatives if applicable. "
        f"Format your response as a numbered list with this structure for each item:\n"
        f"X. **Product Name** - Price: ₦XXX - Discount: XX% - Rating: X.X - Source: [Jumia/Konga]\n"
        f"   - Reason: [Why it’s recommended]\n"
        f"   - Link: <a href='[URL]'>Buy Now</a>\n"
        f"Do not recommend products exceeding the budget of ₦{budget}, but include slightly above-budget items if they have significant discounts."
    )

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
                print(f"LLM attempt {attempt + 1} failed: {str(e)}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise Exception(f"Failed to get LLM response after {max_retries} attempts: {str(e)}")


# Testing
if __name__ == "__main__":
    query = "android phone with good specs for content creation"
    budget = 200000
    num_results = 1 
    source_filter = "All"
    recommendation = get_recommendations(query, budget, num_results, source_filter)
    safe_output = recommendation.encode("ascii", "ignore").decode("ascii")
    print(safe_output) 