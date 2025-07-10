import os
import re
import statistics
from collections import defaultdict, Counter
import difflib
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import matplotlib.colors as mcolors

def read_markdown_file(file_path):
    """Read a markdown file and return its content."""
    with open(file_path, 'r') as file:
        return file.read()

def parse_recipe_markdown(content):
    """Parse markdown content to extract recipe information."""
    # Extract title
    title_match = re.search(r'# (.*)', content)
    title = title_match.group(1) if title_match else "Unknown Recipe"

    # Extract ingredients
    ingredients = []
    ingredients_section = re.search(r'## Ingredients:(.*?)##', content, re.DOTALL)
    if ingredients_section:
        ingredients_text = ingredients_section.group(1)
        # Skip the header rows and extract ingredient rows
        ingredient_rows = re.findall(r'\s*\|\s*([\d\.]+)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|', ingredients_text)
        for amount, unit, name in ingredient_rows:
            ingredients.append({"amount": float(amount), "unit": unit, "name": name.strip()})

    # Extract instructions
    instructions = []
    instructions_section = re.search(r'## Instructions:(.*?)($|\Z)', content, re.DOTALL)
    if instructions_section:
        instructions_text = instructions_section.group(1)
        # Extract numbered instructions
        instruction_matches = re.findall(r'(\d+)\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)', instructions_text, re.DOTALL)
        for _, instruction in instruction_matches:
            instructions.append(instruction.strip())

    return {
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions
    }

def calculate_statistics(recipes):
    """Calculate statistics for all recipes."""
    stats = {}

    # Calculate counts for each recipe
    for file_num, recipe in recipes.items():
        stats[file_num] = {
            "ingredients_count": len(recipe["ingredients"]),
            "instructions_count": len(recipe["instructions"]),
            "ingredients": recipe["ingredients"],
            "instructions": recipe["instructions"]
        }

    # Calculate averages
    all_ingredient_counts = [s["ingredients_count"] for s in stats.values()]
    all_instruction_counts = [s["instructions_count"] for s in stats.values()]

    avg_ingredients = statistics.mean(all_ingredient_counts)
    avg_instructions = statistics.mean(all_instruction_counts)

    # Calculate deviations
    for file_num, stat in stats.items():
        stat["ingredients_deviation"] = ((stat["ingredients_count"] - avg_ingredients) / avg_ingredients) * 100
        stat["instructions_deviation"] = ((stat["instructions_count"] - avg_instructions) / avg_instructions) * 100

        # Calculate overall deviation score (average of absolute deviations)
        stat["deviation_score"] = (abs(stat["ingredients_deviation"]) + abs(stat["instructions_deviation"])) / 2

    return stats, avg_ingredients, avg_instructions

def find_common_deviations(recipes, stats):
    """Find common and egregious deviations in recipes."""
    # Collect all ingredients and instructions
    all_ingredients = []
    all_instructions = []

    for recipe in recipes.values():
        for ingredient in recipe["ingredients"]:
            all_ingredients.append(ingredient["name"])
        all_instructions.extend(recipe["instructions"])

    # Count occurrences
    ingredient_counter = Counter(all_ingredients)
    instruction_counter = Counter(all_instructions)

    # Find most common ingredients and instructions
    common_ingredients = ingredient_counter.most_common(5)
    common_instructions = instruction_counter.most_common(5)

    # Find most egregious deviations (recipes with highest deviation scores)
    egregious_deviations = sorted(stats.items(), key=lambda x: x[1]["deviation_score"], reverse=True)[:5]

    # Find most average recipe (closest to 0% deviation)
    most_average = sorted(stats.items(), key=lambda x: x[1]["deviation_score"])[:1]

    return common_ingredients, common_instructions, egregious_deviations, most_average

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string for embedding in markdown."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_deviation_graph(stats):
    """Create a graph showing deviation amounts for each file."""
    # Sort stats by file number
    sorted_stats = sorted(stats.items())
    file_nums = [f"out_{file_num}" for file_num, _ in sorted_stats]
    deviation_scores = [stat["deviation_score"] for _, stat in sorted_stats]
    ingredient_deviations = [stat["ingredients_deviation"] for _, stat in sorted_stats]
    instruction_deviations = [stat["instructions_deviation"] for _, stat in sorted_stats]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot overall deviation scores
    bars = ax1.bar(range(len(file_nums)), deviation_scores, color='skyblue')
    ax1.set_title('Overall Deviation Scores by Recipe')
    ax1.set_ylabel('Deviation Score (%)')
    ax1.set_xticks([])  # Hide x-axis labels for clarity

    # Highlight top 5 deviations
    top_indices = sorted(range(len(deviation_scores)), key=lambda i: deviation_scores[i], reverse=True)[:5]
    for idx in top_indices:
        bars[idx].set_color('red')
        ax1.text(idx, deviation_scores[idx] + 1, file_nums[idx], ha='center', rotation=90, fontsize=8)

    # Plot ingredient and instruction deviations
    x = np.arange(len(file_nums))
    width = 0.35
    bars1 = ax2.bar(x - width/2, ingredient_deviations, width, label='Ingredients', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, instruction_deviations, width, label='Instructions', color='orange', alpha=0.7)

    ax2.set_title('Ingredient and Instruction Deviations by Recipe')
    ax2.set_ylabel('Deviation (%)')
    ax2.set_xticks([])  # Hide x-axis labels for clarity
    ax2.legend()

    # Add a horizontal line at y=0 for reference
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Highlight the same top 5 deviations
    for idx in top_indices:
        ax2.text(idx, max(ingredient_deviations[idx], instruction_deviations[idx]) + 5, 
                file_nums[idx], ha='center', rotation=90, fontsize=8)

    plt.tight_layout()
    return fig

def create_visual_diff(recipe1, recipe2, title1, title2):
    """Create a visual diff between two recipes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    # Recipe 1 (left side)
    ingredients1 = [ing["name"] for ing in recipe1["ingredients"]]
    instructions1 = recipe1["instructions"]

    # Recipe 2 (right side)
    ingredients2 = [ing["name"] for ing in recipe2["ingredients"]]
    instructions2 = recipe2["instructions"]

    # Create text for ingredients
    ing_text1 = "INGREDIENTS:\n" + "\n".join([f"- {ing}" for ing in ingredients1])
    ing_text2 = "INGREDIENTS:\n" + "\n".join([f"- {ing}" for ing in ingredients2])

    # Create text for instructions (limited to first 5 for clarity)
    inst_text1 = "\nINSTRUCTIONS:\n" + "\n".join([f"{i+1}. {inst[:50]}..." for i, inst in enumerate(instructions1[:5])])
    inst_text2 = "\nINSTRUCTIONS:\n" + "\n".join([f"{i+1}. {inst[:50]}..." for i, inst in enumerate(instructions2[:5])])

    # Combine texts
    text1 = ing_text1 + inst_text1
    text2 = ing_text2 + inst_text2

    # Display texts
    ax1.text(0.05, 0.95, text1, transform=ax1.transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.05, 0.95, text2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # Set titles
    ax1.set_title(title1)
    ax2.set_title(title2)

    # Remove axes
    ax1.axis('off')
    ax2.axis('off')

    plt.tight_layout()
    return fig

def generate_report(stats, avg_ingredients, avg_instructions, common_ingredients, common_instructions, egregious_deviations, most_average, recipes):
    """Generate a markdown report with the statistics and deviations."""
    report = "# Mac & Cheese Recipe Analysis Report\n\n"

    # Create deviation graph
    deviation_graph = create_deviation_graph(stats)
    graph_img = fig_to_base64(deviation_graph)
    plt.close(deviation_graph)  # Close the figure to free memory

    # Add deviation graph to report
    report += "## Recipe Deviation Graph\n\n"
    report += "This graph shows the deviation of each recipe from the average, highlighting where the deviations occur.\n\n"
    report += f"![Deviation Graph](data:image/png;base64,{graph_img})\n\n"

    # Overall statistics
    report += "## Overall Statistics\n\n"
    report += f"- Average number of ingredients: {avg_ingredients:.2f}\n"
    report += f"- Average number of instructions: {avg_instructions:.2f}\n\n"

    # Individual file statistics
    report += "## Individual Recipe Statistics\n\n"
    report += "| File | Ingredients | % Diff | Instructions | % Diff | Overall Deviation |\n"
    report += "|------|-------------|--------|--------------|--------|-------------------|\n"

    for file_num, stat in sorted(stats.items()):
        report += f"| [out_{file_num}.md](out_{file_num}.md) | {stat['ingredients_count']} | {stat['ingredients_deviation']:.2f}% | {stat['instructions_count']} | {stat['instructions_deviation']:.2f}% | {stat['deviation_score']:.2f}% |\n"

    # Most common ingredients
    report += "\n## Top 5 Most Common Ingredients\n\n"
    for ingredient, count in common_ingredients:
        report += f"- **{ingredient}**: appears in {count} recipes\n"

    # Most common instructions
    report += "\n## Top 5 Most Common Instructions\n\n"
    for instruction, count in common_instructions:
        report += f"- **{instruction[:50]}...**: appears in {count} recipes\n"

    # Most egregious deviations with visual diffs
    report += "\n## Top 5 Most Egregious Deviations\n\n"

    # Get the most average recipe for comparison
    avg_file_num, avg_stat = most_average[0]
    avg_recipe = recipes[avg_file_num]

    for file_num, stat in egregious_deviations:
        report += f"### [out_{file_num}.md](out_{file_num}.md) - Deviation Score: {stat['deviation_score']:.2f}%\n\n"
        report += f"- Ingredients: {stat['ingredients_count']} ({stat['ingredients_deviation']:.2f}% from average)\n"
        report += f"- Instructions: {stat['instructions_count']} ({stat['instructions_deviation']:.2f}% from average)\n\n"

        # Create visual diff between this recipe and the most average recipe
        deviant_recipe = recipes[file_num]
        diff_fig = create_visual_diff(
            avg_recipe, 
            deviant_recipe, 
            f"Most Average Recipe (out_{avg_file_num}.md)", 
            f"Deviant Recipe (out_{file_num}.md)"
        )
        diff_img = fig_to_base64(diff_fig)
        plt.close(diff_fig)  # Close the figure to free memory

        # Add visual diff to report
        report += "#### Visual Comparison with Most Average Recipe\n\n"
        report += f"![Visual Diff](data:image/png;base64,{diff_img})\n\n"

        # Create text diff for ingredients
        report += "#### Ingredient Differences\n\n"
        report += "<div style='display: flex;'>\n"
        report += "<div style='flex: 1; background-color: #f0f0f0; padding: 10px; margin-right: 5px;'>\n"
        report += f"**Most Average Recipe (out_{avg_file_num}.md)**\n\n"
        for i, ingredient in enumerate(avg_recipe["ingredients"]):
            report += f"{i+1}. {ingredient['name']}\n"
        report += "</div>\n"
        report += "<div style='flex: 1; background-color: #ffe0e0; padding: 10px; margin-left: 5px;'>\n"
        report += f"**Deviant Recipe (out_{file_num}.md)**\n\n"
        for i, ingredient in enumerate(deviant_recipe["ingredients"]):
            report += f"{i+1}. {ingredient['name']}\n"
        report += "</div>\n"
        report += "</div>\n\n"

        # Create text diff for instructions
        report += "#### Instruction Differences\n\n"
        report += "<div style='display: flex;'>\n"
        report += "<div style='flex: 1; background-color: #f0f0f0; padding: 10px; margin-right: 5px;'>\n"
        report += f"**Most Average Recipe (out_{avg_file_num}.md)**\n\n"
        for i, instruction in enumerate(avg_recipe["instructions"][:5]):  # Limit to first 5 for clarity
            report += f"{i+1}. {instruction[:100]}...\n\n"
        report += "</div>\n"
        report += "<div style='flex: 1; background-color: #ffe0e0; padding: 10px; margin-left: 5px;'>\n"
        report += f"**Deviant Recipe (out_{file_num}.md)**\n\n"
        for i, instruction in enumerate(deviant_recipe["instructions"][:5]):  # Limit to first 5 for clarity
            report += f"{i+1}. {instruction[:100]}...\n\n"
        report += "</div>\n"
        report += "</div>\n\n"

    return report

def main():
    # Directory containing the markdown files
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read and parse all markdown files
    recipes = {}
    for i in range(1, 100):  # Files from out_1.md to out_99.md
        file_path = os.path.join(script_dir, f"out_{i}.md")
        if os.path.exists(file_path):
            content = read_markdown_file(file_path)
            recipes[i] = parse_recipe_markdown(content)

    # Calculate statistics
    stats, avg_ingredients, avg_instructions = calculate_statistics(recipes)

    # Find common and egregious deviations
    common_ingredients, common_instructions, egregious_deviations, most_average = find_common_deviations(recipes, stats)

    # Generate report
    report = generate_report(
        stats, 
        avg_ingredients, 
        avg_instructions, 
        common_ingredients, 
        common_instructions, 
        egregious_deviations,
        most_average,
        recipes
    )

    # Write report to file
    report_path = os.path.join(script_dir, "recipe_analysis_report.md")
    with open(report_path, 'w') as file:
        file.write(report)

    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    main()
