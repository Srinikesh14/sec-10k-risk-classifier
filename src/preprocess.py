import os
import re
import pandas as pd
from datasets import load_dataset

def fetch_and_prepare_data():
    """Stage 1: Load data from Hugging Face via streaming to bypass large downloads."""
    print("--- Starting Stage 1: Data Extraction ---")
    print("Streaming dataset from Hugging Face (Grabbing 1500 rows)...")
    
    # Add streaming=True to avoid downloading the massive underlying files
    dataset = load_dataset("winterForestStump/10-K_sec_filings", split="001", streaming=True)
    
    # Stream exactly 500 rows directly into a list
    data_list = []
    for i, row in enumerate(dataset):
        if i >= 1500:
            break
        data_list.append(row)
        
    df = pd.DataFrame(data_list)
    
    print(f"Successfully grabbed {len(df)} raw documents.")
    return df

def clean_text(text):
    """Cleans up raw text by removing HTML tags, numbers, and extra spaces."""
    if not isinstance(text, str):
        return ""
    
    # 1. Strip HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 3. Remove double/triple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 4. Lowercase everything
    return text.lower()

def extract_section(text, start_keyword, end_keyword):
    """Slices out a specific item/section from the filing text."""
    pattern = rf'(?i){start_keyword}(.*?)(?i){end_keyword}'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else "Section not found"

if __name__ == "__main__":
    import os
    
    # 1. Fetch the data
    df = fetch_and_prepare_data()
    
    print("Cleaning the 'Risk Factors' section...")
    
    # Sometimes a company forgets to include a section, so we fill missing ones with empty text
    df['Risk Factors'] = df['Risk Factors'].fillna("")
    
    # 2. Run our cleaning function directly on the Risk Factors column
    df['cleaned_text'] = df['Risk Factors'].apply(clean_text)
    
    # 3. Let's throw away all the giant columns we don't need so our CSV is lightweight
    final_df = df[['company_name', 'filing_date', 'Risk Factors', 'cleaned_text']]
    
    # 4. Save it!
    os.makedirs('data', exist_ok=True)
    final_df.to_csv('data/processed_data.csv', index=False)
    
    print("Stage 1 Complete! Saved clean data to data/processed_data.csv")