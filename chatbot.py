import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate responses
def generate_response(input_text):
    # Tokenize input with attention mask
    inputs = tokenizer.encode_plus(input_text + tokenizer.eos_token, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Generate response
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode response
    response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Reflective prompts
reflective_prompts = [
    "Can you tell me more about why you felt that way?",
    "What specifically made you feel that way?",
    "How did that make you feel overall?",
    "Why do you think that particular aspect stood out to you?",
    "Can you elaborate on that feeling?"
]

# Function to start the chatbot conversation
def start_chatbot():
    print("Hello! I'm here to help you reflect on your volunteering experiences. You can start by telling me about your recent volunteering activity.")
    initial_input = input("You: ")
    response = generate_response(initial_input)
    print(f"ReflectiveBot: {response}")
    
    while True:
        try:
            for prompt in reflective_prompts:
                user_input = input(f"ReflectiveBot: {prompt}\nYou: ")
                if user_input.lower() in ["stop", "exit", "done"]:
                    print("ReflectiveBot: Thank you for sharing your experiences. Have a great day!")
                    return
                reflective_response = generate_response(user_input)
                print(f"ReflectiveBot: {reflective_response}")
        except (KeyboardInterrupt, EOFError, SystemExit):
            break

if __name__ == "__main__":
    start_chatbot()
