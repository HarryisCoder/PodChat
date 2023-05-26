import openai  # for generating embeddings
import os
import pandas as pd  # for DataFrames to store article sections and embeddings

openai.api_key = os.environ["OPENAI_API_KEY"]

def read_file_with_custom_line_breaker(file_path, line_breaker):
    try:
        with open(file_path, 'r') as file:
            content = file.read().split(line_breaker)
            return [line.strip() for line in content if line.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except IOError:
        print(f"Error reading file: {file_path}")

def calculate_embeddings(strings, embedding_model, batch_size):
    embeddings = []
    for batch_start in range(0, len(strings), batch_size):
        batch_end = batch_start + batch_size
        batch = strings[batch_start:batch_end]
        print(f"Batch {batch_start} to {len(batch)-1}")
        response = openai.Embedding.create(model=embedding_model, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)
    df = pd.DataFrame({"text": strings, "embedding": embeddings})
    return df

# segment show notes into strings
file_path = 'resources/show_notes.txt'
line_breaker = '---'

string_list = read_file_with_custom_line_breaker(file_path, line_breaker)
for i in range(len(string_list)):
    print(f"string {i}: {string_list[i]}\n")

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request
SAVE_PATH = "resources/show_notes_embeddings.csv"
df = calculate_embeddings(string_list, EMBEDDING_MODEL, BATCH_SIZE)
df.to_csv(SAVE_PATH, index=False)