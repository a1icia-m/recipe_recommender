#Import and Set Up Dataset
import pyarrow.parquet as pa
import numpy as np
recipes = pa.read_table(r'data\recipes.parquet')

#Clean dataset
def clean_df(path="data/recipes.parquet", max_rows=None):
    # Load Parquet file
    df = pa.read_table(path).to_pandas()

    if max_rows:
        df = df.head(max_rows)

    # Columns to keep
    keep_cols = [
        "Name", "Description", "RecipeIngredientParts", "RecipeIngredientQuantities",
        "RecipeInstructions", "TotalTime", "Keywords",
        "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
        "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent",
        "ProteinContent"
    ]
    df = df[keep_cols]

    #Drop any rows with missing ingredients or name
    df = df.dropna(subset=["Name", "RecipeIngredientParts"])

    #Convert numpy arrays to lists for easier json output
    df = df[df["RecipeIngredientParts"].apply(lambda x: isinstance(x, (list, np.ndarray)))]
    df["RecipeIngredientParts"] = df["RecipeIngredientParts"].apply(list)


    # Fill nulls in text columns
    text_cols = ["Description", "RecipeInstructions", "Keywords", "TotalTime"]
    for col in text_cols:
        df[col] = df[col].fillna("")

    # Reset index and return
    return df.reset_index(drop=True)

#Test with first 10k rows and output as json
df = clean_df(max_rows = 100000)
print(df.columns) 
print(df.shape)  
df.to_json("data/new_recipe_test_clean.json", orient="records", indent=2)

