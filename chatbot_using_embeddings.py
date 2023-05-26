import ast  # for converting embeddings saved as strings back to arrays
import openai  # for generating embeddings
import os
import pandas as pd  # for DataFrames to store article sections and embeddings
from scipy import spatial  # for calculating vector similarities for search
import tiktoken  # for counting tokens
import streamlit as st


# Web UI
st.title("🎙️仨言俩语AI助手")
st.sidebar.header("使用指南")
st.sidebar.info(
    '''你好👋，我叫利玛逗😆，是播客仨言俩语的AI助手。我可以根据你的提问向你推荐相关的播客节目～
       \n只需要在对话框里输入你的问题然后按回车键就好啦。
       '''
    )

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

#API key
# openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = "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"

# embedding source
embeddings_path = "resources/show_notes_embeddings.csv"
df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df, top_n=5)
    introduction = '请根据以下播客单集简介推荐一期和提问最相关的节目，需要满足以下三个要求：1.如果找到相关节目，请在回答时包含完整的节目标题格式如同“xxx期节目《xxx》”并提供相关内容的时间戳信息 2. 如果找不到答案则以“抱歉，我没有找到直接相关的节目。但是我想为你推荐”作为开头推荐第一个播客单集简介所对应的节目。3. 不要篡改节目标题 4. 不要回答和节目不相关的内容'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\n播客单集简介:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "你作为播客小助手根据播客单集简介帮助回答用户问题"},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


# strings, relatednesses = strings_ranked_by_relatedness("面试指南", df, top_n=3)
# for string, relatedness in zip(strings, relatednesses):
#     print(f"{relatedness=:.3f}"
#     print(string[0:100])

# query = "有没有让人开心的节目？"
# print(f"Q: {query}")
# print(f"A: {ask(query)}")

def main():
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("利玛逗: 客官，今天想听点啥？\n比如你可以问我：\"请推荐一期和面试相关的节目。\"，或者\"有没有让人开心的节目？\"", "")
    if user_query != "":
        # Pass the query to the ChatGPT function
        response = ask(user_query)
        return st.write(f"利玛逗: {response}")

# call the main function
main() 