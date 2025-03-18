import csv
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load and preprocess CSV data
#C:\Users\USER\Desktop\llm_exe\Neural-Drip\data\jumia_and_konga_data2.csv
data_path='C:/Users/USER/Desktop/llm_exe/Neural-Drip/data'
products = []
csv_path = os.path.join(data_path, "jumia_and_konga_data2.csv")
print(f"Attempting to open CSV at: {csv_path}")
try:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"CSV headers: {reader.fieldnames}")
        for i, row in enumerate(reader):
            try:
                safe_row = {k: v.encode("ascii", "ignore").decode("ascii") for k, v in row.items()}
                print(f"Processing row {i}: {safe_row}")

                price_str = row["price"].replace("₦", "").replace("NGN", "").replace(",", "").strip()
                if "-" in price_str:
                    price = float(price_str.split("-")[0].strip())
                else:
                    price = float(price_str)

                rating_str = row.get("ratings", "")
                rating = None
                if rating_str and "out of" in rating_str:
                    rating = float(rating_str.split(" out of ")[0])
                elif rating_str and rating_str.replace(".", "").isdigit():
                    rating = float(rating_str)
                else:
                    rating = 0.0

                discount_str = row.get("discount", "0")
                if discount_str == "No Discount":
                    discount = 0.0
                else:
                    discount = float(discount_str.replace("%", "")) / 100 if discount_str and "%" in discount_str else float(discount_str) / 100 if discount_str.isdigit() else 0.0

                if price < 0 or discount > 1:
                    safe_name = row["product_name"].encode("ascii", "ignore").decode("ascii")
                    print(f"Invalid data for {safe_name}: price={price}, discount={discount}")
                    continue

                # Use the source column directly
                source = row.get("source", "Unknown").capitalize()  # Capitalize for consistency (Jumia, Konga)

                products.append({
                    "product_name": row["product_name"],
                    "price_ngn": price,
                    "discount": discount,
                    "rating": rating,
                    "product_link": row["product_link"],
                    "source": source
                })
                safe_name = row["product_name"].encode("ascii", "ignore").decode("ascii")
                print(f"Successfully processed: {safe_name}")
            except (ValueError, KeyError) as e:
                safe_name = row.get("product_name", "Unknown").encode("ascii", "ignore").decode("ascii")
                safe_error = str(e).encode("ascii", "ignore").decode("ascii")
                print(f"Error processing {safe_name}: {safe_error}")
                continue
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

if not products:
    raise ValueError(f"No valid products processed from {csv_path}. Check CSV content and parsing logic.")

# Prepare data for embedding
texts = [f"{p['product_name']} - ₦{p['price_ngn']} - Discount: {p['discount']*100}% - Rating: {p['rating']} - Source: {p['source']}" for p in products]
ids = [f"prod_{i}" for i in range(len(products))]
metadatas = [
    {"price_ngn": p["price_ngn"], "discount": p["discount"], "rating": p["rating"], "url": p["product_link"], "source": p["source"]}
    for p in products
]

# Validate texts
invalid_texts = [t for t in texts if not isinstance(t, str) or not t.strip()]
if invalid_texts:
    print(f"Found {len(invalid_texts)} invalid texts: {invalid_texts[:5]}")
    raise ValueError("Invalid entries in texts list.")
safe_texts = [t.encode("ascii", "ignore").decode("ascii") for t in texts[:2]]
print(f"Prepared {len(texts)} texts for embedding: {safe_texts}")

# Generate embeddings with batching
batch_size = 160
embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    print(f"Embedding batch {i // batch_size + 1} of {len(texts) // batch_size + 1} ({len(batch)} items)")
    response = client.embeddings.create(input=batch, model="text-embedding-ada-002")
    embeddings.extend([r.embedding for r in response.data])

# Initialize Chroma client
path='C:/Users/USER/Desktop/llm_exe/Neural-Drip/backend/vector_db'
chroma_client = chromadb.PersistentClient(path=path)
chroma_client.delete_collection("products")  # Reset to avoid duplicates
collection = chroma_client.create_collection(name="products")

# Store in Chroma
collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)
safe_final_msg = f"Data embedded and stored! {len(products)} products processed.".encode("ascii", "ignore").decode("ascii")
print(safe_final_msg)