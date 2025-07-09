from openai import OpenAI

import json

# Initialize OpenAI client
client = OpenAI()


def analyze_sentiment(review):
    """
    Analyze the sentiment of a movie review using structured output.
    Returns a dictionary with 'thought' and 'sentiment' keys.
    """
    # TODO: Create a prompt that:
    # 1. Asks for sentiment analysis
    # 2. Specifies the required output format
    #       thought: [analysis]
    #       sentiment: [positive/negative]
    # 3. Includes the review text
    prompt = f"""
Provide me with a sentiment analysis, and sentiment score for the following movie review:

Review: {review}

In your output i want you to do it on two lines, with the first line being the written analysis, the second ONLY the word positive or negative, depending on the sentiment of the movie review.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    # split the response into two lines
    content = response.choices[0].message.content.splitlines()

    # TODO: Parse the response to extract thought and sentiment
    # The response should be in the format:
    # thought: [analysis]
    # sentiment: [positive/negative]
    result = {
        "thought": content[0],
        "sentiment": content[1]
    }

    return result


def main():
    # Test cases
    reviews = [
        "This film shouldn't work at all. It doesn't have much of a story and the whole dial up internet thing is incredibly dated. However Hanks and Ryan sell it beautifully.",
        "The movie was terrible. The acting was wooden, the plot made no sense, and I want my two hours back.",
        "An absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me engaged from start to finish."
    ]

    # Test each review
    for i, review in enumerate(reviews, 0):
        result = analyze_sentiment(review)
        print(f"\nReview {i}:")
        print(f"({review})")
        print(f"Thought: {result['thought']}")
        print(f"Sentiment: {result['sentiment'].lower()}")


if __name__ == "__main__":
    main()
