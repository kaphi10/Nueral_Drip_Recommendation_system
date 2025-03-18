import os
import csv
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load and preprocess CSV data
data_path = "C:/Users/USER/Desktop/llm_exe/Neural-Drip/data"
csv_path = os.path.join(data_path, "jumia_and_konga_data2.csv")

products = []

print(f"Attempting to open CSV at: {csv_path}")
try:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"CSV headers: {reader.fieldnames}")
        
        for i, row in enumerate(reader):
            try:
                # Sanitize data
                safe_row = {k: v.encode("ascii", "ignore").decode("ascii") for k, v in row.items()}
                print(f"Processing row {i}: {safe_row}")

                # Parse price
                price_str = row["price"].replace("₦", "").replace("NGN", "").replace(",", "").strip()
                price = float(price_str.split("-")[0].strip()) if "-" in price_str else float(price_str)

                # Parse rating
                rating_str = row.get("ratings", "").strip()
                if "out of" in rating_str:
                    rating = float(rating_str.split(" out of ")[0])
                elif rating_str.replace(".", "").isdigit():
                    rating = float(rating_str)
                else:
                    rating = 0.0

                # Parse discount
                discount_str = row.get("discount", "0").strip()
                if discount_str == "No Discount":
                    discount = 0.0
                else:
                    discount = float(discount_str.replace("%", "")) / 100 if "%" in discount_str else float(discount_str) / 100 if discount_str.isdigit() else 0.0

                # Validate
                if price < 0 or discount > 1:
                    print(f"Invalid data for {row['product_name']}: price={price}, discount={discount}")
                    continue

                source = row.get("source", "Unknown").capitalize()

                products.append({
                    "product_name": row["product_name"],
                    "price_ngn": price,
                    "discount": discount,
                    "rating": rating,
                    "product_link": row["product_link"],
                    "source": source
                })

                print(f"Successfully processed: {row['product_name']}")

            except (ValueError, KeyError) as e:
                print(f"Error processing {row.get('product_name', 'Unknown')}: {e}")
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

print(f"Prepared {len(texts)} texts for embedding.")

# Generate embeddings with strict batch limit
batch_size = 166  # OpenAI's limit
embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    print(f"Embedding batch {i // batch_size + 1} of {len(texts) // batch_size + 1} ({len(batch)} items)")
    
    response = client.embeddings.create(input=batch, model="text-embedding-ada-002")
    embeddings.extend([r.embedding for r in response.data])

# Initialize ChromaDB client
path = "C:/Users/USER/Desktop/llm_exe/Neural-Drip/backend/vector_db"
chroma_client = chromadb.PersistentClient(path=path)

# Ensure collection doesn't already exist before creating
try:
    chroma_client.delete_collection("products")
except Exception as e:
    print(f"Warning: Couldn't delete existing collection (maybe it doesn't exist yet): {e}")

collection = chroma_client.create_collection(name="products")

# Store embeddings in ChromaDB in batches to avoid exceeding limits
for i in range(0, len(embeddings), batch_size):
    batch_embeds = embeddings[i:i + batch_size]
    batch_texts = texts[i:i + batch_size]
    batch_metas = metadatas[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]

    collection.add(embeddings=batch_embeds, documents=batch_texts, metadatas=batch_metas, ids=batch_ids)

print(f"Data embedded and stored! {len(products)} products processed.")
