import sys
import openai
from dotenv import load_dotenv, find_dotenv


query = sys.argv[1]

CHAT_MODEL_ID = "gpt-35-turbo"
CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID
VECTOR_DB_PATH = "../../data/vectorDB/foodON"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_TEMPERATURE = 0

config_file = os.path.join(os.path.expanduser('~'), '.gpt_config.env')
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = resource_endpoint
openai.api_version = api_version

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

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
node_search_result = vectorstore.similarity_search_with_score(query, k=25)
food_candidates_names = []
food_candidates_ids = []
for item in node_search_result:
    food_candidates_names.append(item[0].page_content)
    food_candidates_ids.append(item[0].metadata["foodON_ID"])

food_candidates_names_str = ", ".join(food_candidates_names)

SYSTEM_PROMPT = """
    You are expert in identifying Food entities. Find the best match for the name of the food given in the Query given below to the options given in the Context provided
"""

enriched_prompt = "Context: "+ food_candidates_names_str + "\n" + "Query: " + query
output = get_GPT_response(enriched_prompt, SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=LLM_TEMPERATURE)

print(output)
