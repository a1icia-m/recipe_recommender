from fastapi import FastAPI, Query
from typing import List
import pandas as pd
from models.recommend import smart_search, apply_nutrition_filters, search, hybrid_nutrition_intents, macros_filter


app = FastAPI()

@app.get("/")
def root():
    return {"message": "It works!"}


@app.get("/search/")
def search_recipes(query: str):

    #Extract ingredients and nutrtion filters from user query 
    ingredients, filters = smart_search(query)   
    tags = hybrid_nutrition_intents(query)  

    #Add any tags missed by semantic/fuzzy search match
    for tag in tags:
        if tag in macros_filter:
            for k, v in macros_filter[tag].items():
                if k not in filters:
                    filters[k] = v
 
    
    #Return top 75 results relevant to ingredients, then apply nutrtion filters and return top 5
    results = search(ingredients, topk=75)
    filtered_results = apply_nutrition_filters(results, filters)

    #Format columns of information to be displayed
    columns = ["Name", "Description", "RecipeIngredientParts", "RecipeIngredientQuantities", "TotalTime", "RecipeInstructions", "Calories", "FatContent", 
               "SaturatedFatContent", "CholesterolContent", "SodiumContent", "CarbohydrateContent", 
               "FiberContent", "SugarContent", "ProteinContent"]

    output = [{col: r.get(col) for col in columns} for r in filtered_results[:5]]
  
    if not filtered_results: 
        return {
        "recipes": [],
        "ingredients": ingredients,
        "filters": filters,
        "message": "No recipes found :( Please try different ingredients or nutritional filters!"
    }


    return {"ingredients": ingredients, "tags": tags, "filters": filters, "recipes": output}





