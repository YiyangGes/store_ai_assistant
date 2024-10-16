import json
from collections import defaultdict
import datetime

# Configure OpenAI KEY
import os
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY") # 'ollama'  
model = "gpt-4o-mini"  # "gpt-4o-mini"
base_url = None

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


def get_completion_from_messages(messages, 
                                 model="gpt-4o-mini", 
                                 temperature=0, 
                                 max_tokens=500):
    '''
    Encapsulate a function to access LLM

    Parameters: 
    messages: This is a list of messages, each message is a dictionary containing role and content. The role can be 'system', 'user' or 'assistant', and the content is the message of the role.
    model: The model to be called, default is gpt-4o-mini (ChatGPT) 
    temperature: This determines the randomness of the model output, default is 0, meaning the output will be very deterministic. Increasing temperature will make the output more random.
    max_tokens: This determines the maximum number of tokens in the model output.
    '''
    response = client.chat.completions.create(
        messages=messages,
        model = model, 
        temperature=temperature, # This determines the randomness of the model's output
        max_tokens=max_tokens, # This determines the maximum number of tokens in the model's output
    )

    return response.choices[0].message.content


# products_file = r'End-to-end-app1\ai_store_assistant_proj\products.json'
products_file = "products.json"
categories_file = 'End-to-end-app1\ai_store_assistant_proj\categories.json'


delimiter = "####"



def get_categories():
    with open(categories_file, 'r') as file:
            categories = json.load(file)
    return categories


# load json from products.json
def get_products():
    with open(products_file, 'r') as file:
        products = json.load(file)
    return products



def get_product_list():
    products = get_products()
    product_list = []
    for product in products.keys():
        product_list.append(product)
    
    return product_list

# reorganize the json
def get_products_and_category():
    products = get_products()
    products_by_category = defaultdict(list)
    for product_name, product_info in products.items():
        category = product_info.get('category')
        if category:
            products_by_category[category].append(product_info.get('name'))
    
    return dict(products_by_category)

# A query, get json entry by product name
def get_product_by_name(name):
    products = get_products()
    return products.get(name, None)


# A query, get json entries by category
def get_products_by_category(category):
    products = get_products()
    return [product for product in products.values() if product["category"] == category]



# Get cate and products info needed from user's original questuib
def find_category_and_product(user_input,products_and_category):
    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    OR
        'products': <a list of products that must be found in the allowed products below>

    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Use your judgement to find best fitted categories using customer service query
    Allowed products: {products_and_category}
    
    """
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"},  
    ]
    return get_completion_from_messages(messages)


def parse_json_string(json_string):
    try:
        # Try to parse the JSON string
        parsed_data = json.loads(json_string)
        
        # Ensure the parsed data is a list of dictionaries
        if not isinstance(parsed_data, list):
            raise ValueError("Expected a list of dictionaries.")
        
        for item in parsed_data:
            if not isinstance(item, dict):
                raise ValueError(f"Expected dictionary but found {type(item).__name__}")
        
        # Optionally, process the data further if needed
        return parsed_data
    
    except json.JSONDecodeError as e:
        # Handle errors related to JSON parsing
        print(f"Error parsing JSON: {e}")
        return None
    except ValueError as e:
        # Handle other possible errors, such as type issues
        print(f"Value error: {e}")
        return None
    except Exception as e:
        # Handle any other unforeseen errors
        print(f"An unexpected error occurred: {e}")
        return None


# parse AI repond
def read_string_to_list(input_string):
    if input_string is None:
        return None

    try:
        input_string = input_string.replace("'", "\"")  # Replace single quotes with double quotes for valid JSON
        data = json.loads(input_string)
        return data
    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
        return None


def generate_output_string(data_list):
    """
    Make the list parsed from the responce of modal into a string that can be used for upcoming chat
    """
    output_string = ""

    if data_list is None:
        return output_string

    for data in data_list:
        try:
            if "products" in data:
                products_list = data["products"]
                for product_name in products_list:
                    product = get_product_by_name(product_name)
                    if product:
                        output_string += json.dumps(product, indent=4) + "\n"
                    else:
                        print(f"Error: Product '{product_name}' not found")
            elif "category" in data:
                category_name = data["category"]
                category_products = get_products_by_category(category_name)
                for product in category_products:
                    output_string += json.dumps(product, indent=4) + "\n"
            else:
                print("Error: Invalid object format")
        except Exception as e:
            print(f"Error: {e}")
        
    return output_string


# Function to log user-AI interaction
def log_interaction(message):
    # A new file is create if the date is different
    date_of_day = datetime.datetime.now().strftime("%Y_%m_%d")
    filename = f"{date_of_day}_log.json"
    # Create a log entry with timestamp, and user input, or AI response, roand metadata
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        timestamp: message
    }
    
    # Read existing log file if it exists, else create a new one
    try:
        with open(filename, 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        logs = []

    # Append the new log entry
    logs.append(log_entry)

    # Write back to the JSON file
    with open(filename, 'w') as file:
        json.dump(logs, file, indent=4)

