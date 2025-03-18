import chromadb

# Connect to the persistent Chroma DB
chroma_client = chromadb.PersistentClient(path="backend/vector_db")
collection = chroma_client.get_collection(name="products")

# Verify contents
items = collection.get()
konga_items = [item for item in items["metadatas"] if item.get("source", "").lower() == "konga" and item["price_ngn"] <= 300000]

# Print with UTF-8 encoding fallback
try:
    print(f"Konga items under ₦300,000: {len(konga_items)}")
    for item in konga_items[:5]:
        print(item)
except UnicodeEncodeError:
    safe_str = f"Konga items under N300,000: {len(konga_items)}"  # Replace ₦ with N
    print(safe_str)
    for item in konga_items[:5]:
        print(str(item).encode("ascii", "ignore").decode("ascii"))