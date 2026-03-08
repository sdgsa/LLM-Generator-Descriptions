import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# Load product data from CSV
df = pd.read_csv(r"/Users/sanjuktaghosh/Desktop/testingtemplate.csv")

# Load pre-trained model and tokenizer from Hugging Face
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_description(product_name, product_category, about_product):
    prompt = (
    f"Write a product description for the following product:\n"
    f"Product Name: {product_name}\n"
    f"Product Category: {product_category}\n"
    f"About the Product: {about_product}\n\n"
    "Example1:\n"
    "Adorable Iwako Japanese Vehicle & Plane Eraser Set! Fun and functional erasers perfect for kids, "
    "students, and collectors. These high-quality erasers feature detailed designs of various vehicles and planes. "
    "Great for school, office, or creative projects. Shop now!\n"
    "Avoid using headers like 'Introduce the Product' or 'Highlight Key Features.' Focus only on the product's benefits "
    "and features in a consumer-friendly tone. Only generate description, no unnecessary details.Keep it concise."
    "Product Description:-"
    )
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=250)
    
    outputs = model.generate(
        inputs, 
        max_length=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.5
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated description

    description = full_output.replace(prompt, '').strip()
    if "Product Description:-" in full_output:
        description = full_output.split("Product Description:")[-1].strip()
    else:
        description = "No description generated."
    
    print("=====!!!!!FULL OUTPUT!!!!!!!========")
    print(full_output)
   
    return description

# Generate descriptions for all products
print("\n------GENERATING DESCRIPTION FOR ALL PRODUCTS-------")
tqdm.pandas()
df["Description"] = df.progress_apply(lambda row: generate_description(row["Product Name"], row["Category"], row["About Product"]), axis=1)

# Save the descriptions back to a new CSV
output_file = "/Users/sanjuktaghosh/Desktop/research/products_with_descriptions.csv"
df.to_csv(output_file, index=False)
print(f"\nDescriptions generated and saved to {output_file}!")
