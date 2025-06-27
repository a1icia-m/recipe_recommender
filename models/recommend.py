import os
import json
import faiss
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from transformers import pipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.nutrition_model import predict_nutrition_intents

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#Set model and file paths
model = SentenceTransformer("all-MiniLM-L6-v2")

recipe_path = r"C:\Users\alici\Desktop\code\misc\recipe_recommender\data\new_recipe_test_clean.json"
embedding_path = r"C:\Users\alici\Desktop\code\misc\recipe_recommender\outputs\recipe_embeddings.npy"
index_path = r"C:\Users\alici\Desktop\code\misc\recipe_recommender\outputs\recipe_index.faiss"
macros_embedding_path = r"C:\Users\alici\Desktop\code\misc\recipe_recommender\outputs\macros_embeddings.npy"

#Load in recipe data
with open(recipe_path, "r", encoding="utf-8") as f:
    recipes = json.load(f)
recipe_text = [", ".join(r["RecipeIngredientParts"]) for r in recipes]

#If first time running, create file, else fetch pre-exisitng embeddings from the file
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

#Embed macro information
macros_filter = {
    "high protein": {"min_protein": 10},
    "low sugar": {"max_sugar": 10},
    "low calorie": {"max_calories": 400},
    "high fiber": {"min_fiber": 8},
    "low fat": {"max_fat": 10},
    "high carb": {"min_carbs": 30}
}
macros_keys = list(macros_filter.keys())

if os.path.exists(macros_embedding_path):
    macros_embeddings = np.load(macros_embedding_path)
else:
    macros_embeddings = model.encode(macros_keys)
    np.save(macros_embedding_path, macros_embeddings)

#Search for most relvant recipes
def search(query_ingredients, topk=50):
    query_vector = model.encode([", ".join(query_ingredients)])[0].astype("float32")
    query_vector = np.array(query_vector).reshape(1, -1)
    distances, indices = index.search(query_vector, topk)
    return [recipes[i] for i in indices[0]]

#Make tokens so it can process wider range of user inputs, not just comma separated ones
def tokenize(query: str):
    # Lowercase and normalize
    query = query.lower()

    # Split on commas, 'and', 'but', etc.
    split_regex = r",|\band\b|\bbut\b|\bor\b|\bwith\b"
    phrases = re.split(split_regex, query)

    # Further split long phrases into individual words (optional)
    tokens = []
    for phrase in phrases:
        cleaned = phrase.strip()
        if cleaned:
            #Include both full phrase and individual words
            tokens.append(cleaned)
            tokens.extend(word for word in cleaned.split() if len(word) > 2)  # skip short stopwords
    return list(set(tokens))  

#Smart search with fuzzy match so it works with typos
def smart_search(user_input, fuzzy_threshold=65, semantic_threshold=0.7):
    # Step 1: Get classifier-based intents
    classifier_intents = predict_nutrition_intents(user_input)

    tokens = tokenize(user_input)
    parsed_filters = {}
    ingredients = []

    classifier_intent_tokens = set()
    for intent in classifier_intents:
        classifier_intent_tokens.update(t.lower() for t in tokenize(intent))

    for token in tokens:
        # Skip if token is already covered by classifier intent
        if any(fuzz.ratio(token, intent_token) >= fuzzy_threshold for intent_token in classifier_intent_tokens):
            continue

        fuzzy_match = next((key for key in macros_filter if fuzz.ratio(token, key) >= fuzzy_threshold), None)
        if fuzzy_match:
            parsed_filters.update(macros_filter[fuzzy_match])
            continue

        query_emb = model.encode([token])[0]
        sim_scores = util.cos_sim(query_emb, macros_embeddings)[0]
        best_idx = sim_scores.argmax()
        if sim_scores[best_idx] >= semantic_threshold:
            matched_macro = macros_keys[best_idx]
            parsed_filters.update(macros_filter[matched_macro])
        else:
            ingredients.append(token)

    # Step 2: Add classifier-detected macros last to avoid redundancy
    for intent in classifier_intents:
        if intent in macros_filter:
            parsed_filters.update(macros_filter[intent])

    return ingredients, parsed_filters

def apply_nutrition_filters(results, filters):
    FIELD_MAP = {
        "sugar": "SugarContent",
        "protein": "ProteinContent",
        "calories": "Calories",
        "fiber": "FiberContent",
        "fat": "FatContent",
        "carbs": "CarbohydrateContent"
    }

    def passes_constraints(recipe):
        for key, value in filters.items():
            base_field = key.replace("min_", "").replace("max_", "")
            recipe_field = FIELD_MAP.get(base_field, base_field)  # default to original

            recipe_val = recipe.get(recipe_field)
            if recipe_val is None:
                return False

            if key.startswith("min_") and recipe_val < value:
                return False
            if key.startswith("max_") and recipe_val > value:
                return False

        return True

    return [r for r in results if passes_constraints(r)]

def fallback_nutrition_intents(query, fuzzy_threshold=65, semantic_threshold=0.7):
    found = set()
    tokens = tokenize(query)

    for token in tokens:
        # Fuzzy match
        match = next((k for k in macros_keys if fuzz.ratio(token, k) >= fuzzy_threshold), None)
        if match:
            found.add(match)
            continue

        # Semantic match
        emb = model.encode([token])[0]
        sim_scores = util.cos_sim(emb, macros_embeddings)[0]
        if sim_scores.max() > semantic_threshold:
            best_match = macros_keys[sim_scores.argmax()]
            found.add(best_match)

    return list(found)

def hybrid_nutrition_intents(query):
    primary = set(predict_nutrition_intents(query))
    fallback = set(fallback_nutrition_intents(query)) if not primary else set()

    return list(primary | fallback)  # use fallback only if classifier fails


