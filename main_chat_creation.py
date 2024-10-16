
from dotenv import load_dotenv
from openai import OpenAI
import os
import utils
import gradio as gr
import time


# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
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


def process_user_message(user_input, all_messages = [], debug=True):
    """
    Preprocess user messages
    
    Parameters:
    user_input : User input
    all_messages : Historical messages
    debug : Whether to enable DEBUG mode, enabled by default
    """
    # Delimiter
    delimiter = "```"

    # Log user input
    # utils.log_interaction(user_input)
    
    # Step 1: Use OpenAI's Moderation API to check if the user input is compliant or an injected Prompt
    response = client.moderations.create(input=user_input.replace(delimiter, ''))
    moderation_output = response.results[0]

    # The input is non-compliant after Moderation API check
    if moderation_output.flagged:
        print("Step 1: Input rejected by Moderation")
        return "Sorry, your request is non-compliant"

    # If DEBUG mode is enabled, print real-time progress
    if debug: print("Step 1: Input passed Moderation check")

    # Step 2: Extract products and corresponding categories 
    category_and_product_response = utils.find_category_and_product(
        user_input, utils.get_products_and_category())
    print(category_and_product_response)
    
    # Convert the extracted string to a list (json)
    category_and_product_list = utils.read_string_to_list(category_and_product_response)
    # print(category_and_product_list)
    
    if debug: print("\nStep 2: Extracted product list")

    # Step 3: Find corresponding product information
    product_information = utils.generate_output_string(category_and_product_list)
    if debug: print("\nStep 3: Found information for extracted products")
    print(product_information)

    # Step 4: Generate answer based on information
    # Ini sys message
    system_message = f"""
    You are a customer service assistant for a large electronic store. \
    Respond in a friendly and helpful tone, with concise answers only based on relevant product infomation, and previous chat history\
    Don't use product information that is not related to the user question.
    - (e.g., If user asked about TVs, don't provide home theater information.)
    Make sure to ask the user relevant follow-up questions.
    Do not make up any product that is not provided in **relevant product information** section, refer to relevant product information before reply
    Reject politely if user message is not relevent to the store service
    """
    # Insert message
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]

    utils.log_interaction(messages[1])
    if product_information:
        utils.log_interaction(messages[2])

    final_response = get_completion_from_messages(all_messages + messages)
    if debug:print("\nStep 4: Generated user answer")
    # print(final_response)

    # Add this round of information to historical messages
    all_messages = all_messages + messages[1:]
    print(all_messages)

    # Step 5: Check if the output is compliant based on Moderation API
    response = client.moderations.create(input=final_response)
    moderation_output = response.results[0]

    # Output is non-compliant
    if moderation_output.flagged:
        if debug: print("Step 5: Output rejected by Moderation")
        return "Sorry, we cannot provide that information"

    if debug: print("Step 5: Output passed Moderation check")

    # Step 6: Model checks if the user's question is well answered
    user_message = f"""
    Customer message: {delimiter}{user_input}{delimiter}
    Agent response: {delimiter}{final_response}{delimiter}

    Does the response sufficiently answer the question?
    Answer "Y" for yes, "N" for no
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    # Request model to evaluate the answer
    evaluation_response = get_completion_from_messages(messages)
    if debug: print("Step 6: Model evaluated the answer")

    # Step 7: If evaluated as Y, output the answer; if evaluated as N, feedback that the answer will be manually corrected
    if "Y" in evaluation_response:  # Use 'in' to avoid the model possibly generating Yes
        if debug: print("Step 7: Model approved the answer.")
        utils.log_interaction({'role':'assistant', 'content':f"{final_response}"})
        return final_response, all_messages
    else:
        if debug: print("Step 7: Model disapproved the answer.")
        neg_str = "I apologize, but I cannot provide the information you need. I will transfer you to a human customer service representative for further assistance."
        utils.log_interaction({'role':'assistant', 'content':f"{final_response}"})
        return neg_str, all_messages

# user_input = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also what tell me about your tvs"


# mainly for keeping track of chat history (that is not seen by user)
# the chat_history paramter is for interface
all_m = []

    
def respond(message, chat_history):
    global all_m

    response, all_m = process_user_message(message, all_m)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    all_m.append({'role':'assistant', 'content':f"{response}"})

    time.sleep(2)
    return "", chat_history

# examples = [["I’m looking for a new smartphone. Can you suggest the best options?",[]], 
#             ["Is the SmartX MiniPhone available in 64GB?",[]],
#             ["What are some available Laptops",[]],
#             ["Does this MobiTech PowerCase come with warranty",[]]]

# create gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>Welcome to <strong>Whiteforwhite Electronic Store</strong>!</h1>")  # Centered title
    gr.HTML("<h2 style='text-align: center;'>Start to chat with me for any product informations!</h2>")  # Centered title        
    gr.Markdown(
        """
        ### I can:
        * help with product inqueries
        * give recommadations
        * tell you about general product informations
        * provide helpful comparisons
        """
    )
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Ask me about product informations~")
    button = gr.Button(value="Submit")
    # clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    button.click(respond, [msg, chatbot], [msg, chatbot])
    # gr.Examples(
    # fn=respond,
    # inputs=[msg, chatbot],
    # outputs=[msg, chatbot],
    # examples = examples)

demo.launch()
