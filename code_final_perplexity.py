import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import textstat
from tqdm import tqdm
import spacy
import re

# Load the dataset
df = pd.read_csv(r'/Users/sanjuktaghosh/Desktop/research/Template_HUMAN_AI_COMPLETE.csv')

# Initialize models
semantic_model = SentenceTransformer('all-mpnet-base-v2')
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",device =0)
nlp = spacy.load("en_core_web_sm")

def semantic_similarity(text1, text2):
    embeddings = semantic_model.encode([text1, text2])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]  # Limit input to 512 tokens
    return result['label'], result['score']

def analyze_readability(text):
    return textstat.flesch_kincaid_grade(text)

def analyze_persuasiveness(text):
    doc = nlp(text)
    persuasive_words = [
    'powerful', 'effortless', 'premium', 'durable', 'innovative', 
    'advanced', 'effective', 'comfortable', 'reliable', 'proven', 
    'limited', 'exclusive', 'instant', 'hurry', 'act now', 
    'deadline', 'final', 'today', 'rush', 'last chance', 
    'guaranteed', 'certified', 'trusted', 'endorsed', 'safe', 
    'recommended', 'authentic', 'award-winning', 'risk-free', 'approved', 
    'love', 'joy', 'unforgettable', 'imagine', 'delight', 
    'transform', 'thrill', 'happiness', 'inspire', 'surprise', 
    'free', 'bonus', 'affordable', 'save', 'bargain', 
    'complimentary', 'worthwhile', 'cost-effective', 'extra', 'valuable',
    'discover', 'unlock', 'elevate', 'achieve', 'dominate', 
    'you deserve', 'tailored', 'just for you', 'personalized', 'yours', 
    'extraordinary', 'luxurious', 'exceptional', 'captivating', 'mesmerizing', 
    'limited supply', 'act fast', 'don\'t miss out', 'only available until', 'time-sensitive', 
    'backed by science', 'made with care', 'sustainable', 'transparent', 'awarded', 
    'exclusive deal', 'reduced price', 'clearance', 'lowest price', 'best value'
    ]
    persuasive_count = sum(1 for token in doc if token.lemma_.lower() in persuasive_words)
    return persuasive_count / len(doc)

def analyze_seo_optimization(text, category):
    if pd.isna(category) or not isinstance(category, str):
        return 0  # Return 0 if category is not a string or is NaN
    keywords = category.lower().split()
    doc = nlp(text.lower())
    keyword_count = sum(1 for token in doc if token.text in keywords)
    return keyword_count / len(doc) if len(doc) > 0 else 0

def analyze_clarity(text):
    doc = nlp(text)
    if len([token for token in doc if token.is_alpha]) == 0:
        return 0
    else:    
        avg_word_length = sum(len(token.text) for token in doc if token.is_alpha) / len([token for token in doc if token.is_alpha])
        if avg_word_length == 0:
            return 0
        else:
            return 1 / avg_word_length  # Higher score for shorter average word length

def analyze_emotional_appeal(text):
    doc = nlp(text)
    emotion_words = ['love', 'happy', 'excited', 'amazing', 'wonderful', 'delighted', 'thrilled']
    emotion_count = sum(1 for token in doc if token.lemma_.lower() in emotion_words)
    return emotion_count / len(doc)

def analyze_call_to_action(text):
    cta_patterns = ['buy now', 'shop now', 'get yours', 'order today', 'learn more', 'discover']
    cta_count = sum(1 for pattern in cta_patterns if pattern in text.lower())
    return cta_count

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    description = row['Description']
    sentiment_label, sentiment_score = analyze_sentiment(description)
    
    results.append({
        'Uniq Id': row['Uniq Id'],
        'Product Name': row['Product Name'],
        'Category': row['Category'],
        'AI or Human': row['AI or Human Generated'],
        'Description': row['Description'],
        'Sentiment': sentiment_label,
        'Sentiment Score': sentiment_score,
        'Readability Score': analyze_readability(description),
        'Persuasiveness Score': analyze_persuasiveness(description),
        'SEO Optimization Score': analyze_seo_optimization(description, row['Category']),
        'Clarity Score': analyze_clarity(description),
        'Emotional Appeal Score': analyze_emotional_appeal(description),
        'Call-to-Action Score': analyze_call_to_action(description)
    })

results_df = pd.DataFrame(results)

# Define categories for analysis
categories = [
    "Human Generated",
    "AI Generated - GPT2",
    "AI Generated - ChatGPT4 manual",
    "AI Generated - Gemma",
    "AI Generated - Gemma -Sample",
    "AI Generated - GPT2 - Sample",
    "AI Generated - LLAMA - Sample",
    "AI Generated-LLama"
]

# Analyze results for each category
print("\nComparison of Different Generation Methods:")
comparison_results = []

for category in categories:
    category_results = results_df[results_df['AI or Human'].str.contains(category, case=False, na=False)]
    
    if len(category_results) > 0:
        print(f"\n{category}:")
        for metric in ['Sentiment Score', 'Readability Score', 'Persuasiveness Score', 'SEO Optimization Score', 'Clarity Score', 'Emotional Appeal Score', 'Call-to-Action Score']:
            avg_score = category_results[metric].mean()
            print(f"  {metric}: {avg_score:.4f}")
            comparison_results.append({'Metric': f"{category} - {metric}", 'Value': f"{avg_score:.4f}"})

# Calculate pairwise semantic similarities
print("\nPairwise Semantic Similarities:")
for i in range(len(categories)):
    for j in range(i+1, len(categories)):
        cat1_descriptions = df[df['AI or Human Generated'].str.contains(categories[i], case=False, na=False)]['Description'].tolist()
        cat2_descriptions = df[df['AI or Human Generated'].str.contains(categories[j], case=False, na=False)]['Description'].tolist()
        
        if cat1_descriptions and cat2_descriptions:
            similarities = [semantic_similarity(desc1, desc2) for desc1, desc2 in zip(cat1_descriptions, cat2_descriptions)]
            avg_similarity = np.mean(similarities)
            print(f"{categories[i]} vs {categories[j]}: {avg_similarity:.4f}")
            comparison_results.append({'Metric': f"Semantic Similarity - {categories[i]} vs {categories[j]}", 'Value': f"{avg_similarity:.4f}"})

# Create a DataFrame for analysis results
analysis_results = pd.DataFrame(comparison_results)

# Concatenate the original results with the analysis results
final_results = pd.concat([results_df, analysis_results], axis=1)

# Save the final results
output_file = r'/Users/sanjuktaghosh/Desktop/research/analysis_results.csv'
final_results.to_csv(output_file, index=False)
print(f"\nDetailed results and analysis saved to '{output_file}'")
