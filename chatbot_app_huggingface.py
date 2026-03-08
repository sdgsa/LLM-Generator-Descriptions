import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load product data from CSV
df = pd.read_csv(r"/Users/sanjuktaghosh/Desktop/research/products.csv")

# Load pre-trained model and tokenizer from Hugging Face
model = GPT2LMHeadModel.from_pretrained("gpt2")  # You can use GPT-2 or GPT-Neo for a larger model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to generate product descriptions
def generate_description(product_name, product_category,about_product):
    prompt = f"Write a product description for {product_name} based on Product Name {product_name}, Product Category {product_category} and About the product {about_product}."
    
    # Encode the prompt text into tensor format
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text using the model
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
    # Decode the generated output and clean the response
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt text from the output (the first line)
    description = description.replace(prompt, '').strip()
    
    return description

# Generate descriptions for all products
df["Description"] = df.apply(lambda row: generate_description(row["Product Name"], row["Category"], row["About Product"), axis=1)

# Save the descriptions back to a new CSV
output_file = "/Users/sanjuktaghosh/Desktop/research/products_with_descriptions.csv"
df.to_csv(output_file, index=False)
print(f"Descriptions generated and saved to {output_file}!")
