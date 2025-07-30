import json
import os
from openai import OpenAI
from langsmith.evaluation import evaluate

# Initialize the OpenAI client
client = OpenAI()

from langsmith import Client

langsmith_client = Client()

# Dataset name in LangSmith (already uploaded)
dataset_name = "news_dataset_class"

# System prompt for extraction
SYSTEM_PROMPT = """Extract information from the news into a dictionary. 
The dictionary keys are company_name, date_of_transaction, amount, product_service, location. 
Make sure the output is in proper JSON with double quotes around the keys and values. 
For all dates, use the following format: mm-dd-yyyy."""

def make_call_to_llm(input):
    # Extract the input content from the dataset item
    user_content = input["news"] if isinstance(input, dict) else input
    
    # Create the message array for the API call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0
    )
    
    # Extract the output from the response
    output = response.choices[0].message.content
    
    return {"output": output}

# llm_result is the result of the make_call_to_llm call
# It contains the model's response that we'll compare against the expected output
def perform_eval(llm_result, dataset_item):
    try:
        # Parse the model's output
        llm_output = json.loads(llm_result['output'])
        
        # Parse the expected output
        expected_output = json.loads(dataset_item['output'])
        
        # Extract score from response
        # For a simpler implementation, let's manually calculate the score
        total_keys = len(expected_output)
        correct_keys = sum(1 for key in expected_output if key in llm_output and llm_output[key] == expected_output[key])
        score = correct_keys / total_keys if total_keys > 0 else 0
        
        return {"score": score}
    except json.JSONDecodeError:
        # Handle the case where JSON parsing fails
        return {"score": 0.0}

# Evaluate the target task
# TODO: Implement the evaluate function to run the evaluation
# See https://docs.smith.langchain.com/evaluation for reference and examples
# This should evaluate make_call_to_llm against the dataset_name using perform_eval
# and create an experiment with the prefix "news_extraction_homework"



if __name__ == "__main__":
    # loop through the csv dataset and call the make_call_to_llm function
    # and print the results
    dataset_path = "news_dataset.csv"  # Replace with your dataset path
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as file:
            # load dataset from CSV
            import csv
            dataset = [row for row in csv.DictReader(file)]
        # Iterate through the dataset and call the LLM
        print("Starting evaluation of the dataset...")
         # Call the LLM for each item in the dataset
         # and print the input and output
        print(f"Loaded {len(dataset)} items from the dataset.")
        for item in dataset:
            result = make_call_to_llm(item)
            print(f"Input: {item['news']}\nOutput: {result['output']}\n")
            score = langsmith_client.evaluate(
                make_call_to_llm,
                data=dataset_name,
                evaluators=[
                    perform_eval
                ],
                experiment_prefix="news_extraction_homework",
                max_concurrency=2,
            )





    else:
        print(f"Dataset file {dataset_path} does not exist.")