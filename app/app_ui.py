import streamlit as st
import requests

st.title("🍽️ Recipe Recommender")

query = st.text_input("Enter ingredients and nutrition goals (optional)", "i.e. egg, avocado, high protein, low sugar")

if st.button("Search"):
    # Call FastAPI recommend endpoint
    try:
        response = requests.get("http://127.0.0.1:8000/search/", params={"query": query})
        response.raise_for_status()
        data = response.json()
        
        # Nutrition fields to display for each recipe
        nutrition_fields = [
            "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
        "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent",
        "ProteinContent"
        ]

        #Display Error message if no relevant recipes found
        if not data['recipes']:
            st.warning(data.get("message", "No recipes found."))
        else:
            st.subheader("Top Recipes")
            #Display most relevant recipes and their metadata
            for recipe in data['recipes']:
                st.markdown(f"### {recipe['Name']}")
                st.write(recipe["Description"])
                st.write("Time:", recipe["TotalTime"])

                #Display ingredient and corresponding quantity
                ingredients_with_quantities = [f"{quantity} {ingredient}" for ingredient, quantity in zip(recipe["RecipeIngredientParts"], recipe["RecipeIngredientQuantities"])]
                st.write("Ingredients:")
                for item in ingredients_with_quantities:
                    st.write(f"- {item}") 

                st.write("Recipe:")
                for item in recipe["RecipeInstructions"]:
                    st.write(f"-{item}") 

                # Display all nutrition information
                nutrient_display = []
                for field in nutrition_fields:
                    if field in recipe:
                        display_name = field.replace("Content", "")
                        is_filtered = any(f.lower() in field.lower() for f in data['filters'])
                        # Highlight filtered fields from user's query with *
                        display_value = f"{display_name}: {recipe[field]} {'*' if is_filtered else ''}"
                        nutrient_display.append(display_value)
                
                st.write(" | ".join(nutrient_display))
                st.markdown("---")

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")