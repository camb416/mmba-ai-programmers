from openai import OpenAI
import faiss
import numpy

# Example documents about movies
texts = [
    "The Godfather is a classic mafia crime drama",
    "Inception explores dreams within dreams",
    "The Shawshank Redemption is a story about hope and friendship",
]



def get_embedding(text):
    # Documentation: https://platform.openai.com/docs/guides/embeddings
    print(" --- --- Embedding Generation --- ---")
    # 1. Create an OpenAI client
    client = OpenAI()

    print(f"Sending the text `{text}` to \"text-embedding-3-small\" model for embedding generation...")
    # 2. Make an API call to generate embeddings using the text-embedding-3-small model
    response  = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    print(f"The vector response: {response.data[0].embedding}\n")

    # 3. Return the embedding vector from the response

    # Placeholder for the actual implementation
    return response.data[0].embedding

embeddings = []
for text in texts:
    embedding = get_embedding(text)
    embeddings.append(embedding)


dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

index.add(numpy.array(embeddings, dtype='float32'))

query = 'Tell me about a prison movie'


query_embedding = get_embedding(query)
distances, indicies = index.search(numpy.array([query_embedding], dtype='float32'), 3)

closest_distance = 999.0
closest_text_id = -1

print(" --- --- Query Processing --- ---")


for i in range(3):
    print(f"Match {i+1}, Distance: {distances[0][i]:.4f}")
    print(texts[indicies[0][i]])
    if distances[0][i] < closest_distance:
        closest_distance = distances[0][i]
        closest_text_id = indicies[0][i]

print("\n --- --- Results --- ---")
if closest_text_id >= 0:
    print(f"The closest match to the query is: `{texts[closest_text_id]}` with a distance of {closest_distance:.4f}.")