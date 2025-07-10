from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from pprint import pprint
import os
import sys

# Model for Receipe Ingredients
class Ingredient(BaseModel):
    amount: float = Field(description="Quantity of the ingredient", default=0.0)
    unit: str = Field(description="Unit of measurement", default="")
    name: str = Field(description="Name of ingredient", default="")

# Model for Recipe. Contains title, array of Ingredients, and array of text instructions.
class Recipe(BaseModel):
    """
    Use this model when working with complete cooking recipes.
    """
    title: str = Field(description="Name of the recipe")
    ingredients: List[Ingredient] = Field(description="List of ingredients needed for the recipe")
    instructions: List[str] = Field(description="Step-by-step instructions to prepare the recipe")

# Makes the OpenAI call to convert recipe text to a structured Recipe object.
def get_recipe_from_text(recipe_text: str) -> Recipe:
    """
    Convert recipe text into a structured Recipe object using OpenAI.
    """
    client = OpenAI()

    # Make the API call
    response = client.responses.parse(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {"role": "user", "content": f"Convert this recipe into the specified format:\n\n{recipe_text}"}
        ],
        text_format=Recipe
    )

    return response.output_parsed

# Function to print the recipe as markdown. Optionally saves to a file.
def print_recipe(recipe: Recipe, saveAsFile: bool = False, filename: str = 'out.md'):
    orig_stdout = sys.stdout
    if saveAsFile:
        f = open(filename, 'w')
        sys.stdout = f

    print(f"# {recipe.title}")
    print(f"\n## Ingredients:")
    print(f"|{'Amount':<10} | {'Unit':<10} | {'Name':<30}|")
    print(f"|{'-' * 10}|{'-' * 10}|{'-' * 30}|")
    for i in recipe.ingredients:
        print(f" | {i.amount} |  {i.unit} | {i.name} |")
    print(f"\n## Instructions:")
    for idx, i in enumerate(recipe.instructions):
        print(f"{idx + 1}. {i}")

    if saveAsFile:
        sys.stdout = orig_stdout
        f.close()

if __name__ == "__main__":
    # Read recipe text from file

    script_dir = os.path.dirname(os.path.abspath(__file__))
    recipe_path = os.path.join(script_dir, "mac_and_cheese_recipe.txt")
    with open(recipe_path, "r") as file:
        recipe_text = file.read()

    # change this to the number of times you want to run the recipe generation
    # This is useful to see if the output is different each time (spoiler alert: it is)

    num_iterations = 1
    for i in range(0,num_iterations):
        # Get structured recipe
        print(f"Processing recipe {i+1}...")
        recipe = get_recipe_from_text(recipe_text)
        print_recipe(recipe, True, f"recipe_out_{i}.md")
