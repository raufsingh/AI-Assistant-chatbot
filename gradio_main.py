import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Hugging Face API Token (Replace with your own key)
hf_token = "hfapikey" 

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    use_auth_token=hf_token, 
    trust_remote_code=True, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Function to generate answer
def generate_answer(question):
    """
    Generates an answer based on the input question using the Phi-1.5 model.
    
    Parameters:
        question (str): The user's question.
    
    Returns:
        str: The AI-generated answer.
    """
    prompt = f"Answer the following question: {question}"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    output = model.generate(input_ids, 
                            max_new_tokens=100,  
                            do_sample=True, 
                            top_k=50, 
                            top_p=0.9, 
                            temperature=0.7)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.strip()

# Gradio chatbot interface
interface = gr.Interface(
    fn=generate_answer,
    inputs="text",
    outputs="text",
    title="I am your AI Health Assistant 🏥",
    description="Ask general health-related questions to the AI Bot."
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch()
