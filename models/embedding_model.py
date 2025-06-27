import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
recipe_path = "data/recipe_test_clean.json"
embedding_path = "outputs/recipe_embeddings.npy"
index_path = "outputs/recipe_index.faiss"
macros_embedding_path = "outputs/macros_embeddings.npy"

# Load recipes
with open(recipe_path, "r", encoding="utf-8") as f:
    recipes = json.load(f)

# Extract recipe text
recipe_text = [", ".join(r["RecipeIngredientParts"]) for r in recipes]

# Generate or load recipe embeddings and FAISS index
if os.path.exists(embedding_path) and os.path.exists(index_path):
    embeddings = np.load(embedding_path)
    index = faiss.read_index(index_path)
else:
    embeddings = model.encode(recipe_text, batch_size=256, show_progress_bar=True).astype("float32")
    np.save(embedding_path, embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

# Nutrition filters
macros_filter = {
    "high protein": {"min_protein": 10},
    "low sugar": {"max_sugar": 10},
    "low calorie": {"max_calories": 400},
    "high fiber": {"min_fiber": 5},
    "low fat": {"max_fat": 10},
    "high carb": {"min_carbs": 30}
}
macros_keys = list(macros_filter.keys())

# Load or embed nutrition filters
if os.path.exists(macros_embedding_path):
    macros_embeddings = np.load(macros_embedding_path)
else:
    macros_embeddings = model.encode(macros_keys)
    np.save(macros_embedding_path, macros_embeddings)

