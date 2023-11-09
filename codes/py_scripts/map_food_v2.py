import sys
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import json
import numpy as np
import os


query = sys.argv[1]

CHAT_MODEL_ID = "gpt-35-turbo"
CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID
VECTOR_DB_PATH = "../../data/vectorDB/foodON"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_TEMPERATURE = 0
DIRECT_MATCH_THRESH = 0.9

config_file = os.path.join(os.path.expanduser('~'), '.gpt_config.env')
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = resource_endpoint
openai.api_version = api_version



def main():
    recognize_food_list = recognize_food(query)    
    if not recognize_food_list:
        output_2 = no_match_resp(query)
    recognize_food_list = [item for item in recognize_food_list if item != query]
    vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
    direct_match = find_food_match(query, vectorstore)
    try:
        if direct_match["confidence_score"] > DIRECT_MATCH_THRESH:
            final_output = direct_match
        else:
            final_output = []
            for item in recognize_food_list:
                final_output.append(find_food_match(item, vectorstore))                            
    except:
        final_output = no_match_resp(query)                
    print(json.dumps(final_output, indent=4))


def recognize_food(input_query):
    FOOD_RECOGNITION_SYSTEM_PROMPT = """
        Find the food items or beverages from the given Query at the end. If you find multiple food items, then make sure to give their components and sensible combinations too. 
        Example 1: 
        Query : "Rice with vegetables", 
        recognized food list : ["Rice", "vegetables", "Rice with vegetables", "vegetables with Rice"]
        Example 2:
        Query : "Sugar, powdered"
        recognized food list : ["Sugar", "Sugar, powdered", "powdered Sugar"]
        Example 3:
        Query : "Burritto"
        recognized food list : ["Burritto"]
        Give the output in JSON format with following syntax:
        {
        query : <given query>,
        recognized_food : <recognized food list>
        }
        If you do not find any food or beverage from the query, give empty list as the value for the key "recognized_food" in the JSON ouput
    """
    food_recognition_prompt = "Query: " + input_query
    food_recognition_output = get_GPT_response(food_recognition_prompt, FOOD_RECOGNITION_SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=LLM_TEMPERATURE)
    food_recognition_output_dict = json.loads(food_recognition_output)
    if len(food_recognition_output_dict["recognized_food"]) == 0:
        return None
    else:
        return food_recognition_output_dict["recognized_food"]
    
    
def find_food_match(input_query, vectorstore):
    node_search_result = vectorstore.similarity_search_with_score(input_query, k=25)
    food_candidates_names = []
    food_candidates_id_dict = {}
    score_match_dict = {}
    for item in node_search_result:
        if item[-1] < 1:
            food_candidates_names.append(item[0].page_content)
            food_candidates_id_dict[item[0].page_content] = item[0].metadata["foodON_ID"]
            score_match_dict[item[0].page_content] = 1 - item[-1]
        else:
            break
    if len(food_candidates_names) != 0:
        food_candidates_names_str = ", ".join(food_candidates_names)
        FOOD_MATCH_SYSTEM_PROMPT = """
            You are expert in identifying Food entities. Which options given in the context match best with the given food name in the Query. Also, provide a confidence score, between 0 and 1, for the best match. Provide the output in JSON format as given below:
            {{
                "query" : <given name>
                "best match" : <match found>
                "confidence" : <confidence score>
            }}
        """
        enriched_prompt = "Context: "+ food_candidates_names_str + "\n" + "Query: " + input_query
        output = get_GPT_response(enriched_prompt, FOOD_MATCH_SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=LLM_TEMPERATURE)
        output_dict = json.loads(output)
        best_match = output_dict["best match"]
        try:
            best_match_id = food_candidates_id_dict[best_match]
            best_match_vector_similarity = score_match_dict[best_match]
            best_match_llm_confidence = output_dict["confidence"]
            best_match_final_confidence = np.mean([best_match_vector_similarity, best_match_llm_confidence])
            output_2 = {
                "query" : query,
                "best_foodON_match" : best_match,
                "best_foodON_match_id" : best_match_id,
                "confidence_score" : best_match_final_confidence
            }
        except:
            output_2 = no_match_resp(query)
    else:
        output_2 = no_match_resp(query)
    return output_2
        
            
           

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    try:
        response = openai.ChatCompletion.create(
            temperature=temperature, 
            deployment_id=chat_deployment_id,
            model=chat_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ]
        )
        if 'choices' in response \
        and isinstance(response['choices'], list) \
        and len(response) >= 0 \
        and 'message' in response['choices'][0] \
        and 'content' in response['choices'][0]['message']:
            return response['choices'][0]['message']['content']
        else:
            return 'Unexpected response'
    except:
        return None
    
def no_match_resp(input_query):
    output = {
        "query" : input_query,
        "best_foodON_match" : '',
        "best_foodON_match_id" : '',
        "confidence_score" : ''
    }
    return output
    

    
    
if __name__ == "__main__":
    main()
