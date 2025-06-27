import pandas as pd
import random
# Generate csv for training classifier to enhance nutrtion intent detection
# Labels and their variants

label_variants = {
    "Calories": [
        "low calorie", "high calorie", "reduce calories", "calorie-conscious", 
        "not too many calories", "calorie-dense", "light on calories", 
        "watch my calories", "low-cal", "high-cal"
    ],
    "FatContent": [
        "low fat", "high fat", "reduce fat", "fat-free", 
        "less fat", "fatty", "light on fat", "no fat", 
        "low-fat", "high-fat"
    ],
    "SaturatedFatContent": [
        "low saturated fat", "high saturated fat", "reduce saturated fat", 
        "saturated fat-free", "less saturated fat", "rich in saturated fat", 
        "watch saturated fat", "no saturated fat"
    ],
    "CholesterolContent": [
        "low cholesterol", "high cholesterol", "reduce cholesterol", 
        "cholesterol-free", "less cholesterol", "rich in cholesterol", 
        "watch my cholesterol", "no cholesterol"
    ],
    "SodiumContent": [
        "low sodium", "high sodium", "reduce sodium", "sodium-free", 
        "less sodium", "salty", "watch my sodium", "no salt", 
        "low-salt", "high-salt"
    ],
    "CarbohydrateContent": [
        "low carb", "high carb", "reduce carbs", "carb-free", 
        "less carbohydrates", "carb-heavy", "watch my carbs", 
        "no carbs", "low-carb", "high-carb"
    ],
    "FiberContent": [
        "high fiber", "low fiber", "fiber-rich", "increase fiber", 
        "fiber-free", "less fiber", "good source of fiber", 
        "no fiber", "dietary fiber"
    ],
    "SugarContent": [
        "low sugar", "high sugar", "reduce sugar", "sugar-free", 
        "less sweet", "sugary", "watch my sugar", "no sugar", 
        "low-sugar", "high-sugar"
    ]
}

# Generate 100 examples per label
queries = []
labels = []
for label, variants in label_variants.items():
    for _ in range(100):
        # Randomly select a variant and modify it
        base = random.choice(variants)
        query = base
        
        # Add noise/paraphrasing (optional)
        if random.random() > 0.5:
            query = f"{query} {random.choice(['recipes', 'meals', 'food', 'dishes', 'options'])}"
        if random.random() > 0.7:
            query = query.replace("low", "not much").replace("high", "lots of")
        
        queries.append(query)
        labels.append(label)

# Create DataFrame and save
df = pd.DataFrame({"Query": queries, "Label": labels})
df.to_csv(r"C:\Users\alici\Desktop\code\misc\recipe_recommender\data\nutrition_intent_dataset.csv", index=False)
print("Dataset saved!")