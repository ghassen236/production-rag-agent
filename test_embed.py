from data_loader import embed_texts

result = embed_texts(["This is a test sentence"])
print(f"Embedding dimension: {len(result[0])}")
print(f"First 3 values: {result[0][:3]}")