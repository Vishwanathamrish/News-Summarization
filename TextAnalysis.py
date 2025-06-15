import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import re
import textstat
import requests
from bs4 import BeautifulSoup

input_df = pd.read_excel("Input.xlsx")


print(input_df.head())



def extract_article(url):
    """Extract the title and main article text from a given URL."""
    try:
        response = requests.get(url, timeout=10)  
        print(response.status_code)
        response.raise_for_status()  

        soup = BeautifulSoup(response.text, 'html.parser')

       
        title = soup.find('h1').text.strip() if soup.find('h1') else "No Title Found"
        paragraphs = soup.find_all('p')  
        content = ' '.join([p.text.strip() for p in paragraphs]) if paragraphs else "No Content Found"

        # print(content)
        return title, content
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None, None



output_folder = "Extracted_Articles"
os.makedirs(output_folder, exist_ok=True)

for index, row in input_df.iterrows():
    url_id = row["URL_ID"]
    url = row["URL"]
    
    print(f"Processing URL_ID: {url_id} - {url}")
    
    title, text = extract_article(url)

    if title and text:
        
        file_path = os.path.join(output_folder, f"{url_id}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(title + "\n\n" + text)
        
        print(f"Saved: {file_path}")
    else:
        print(f"Skipping {url_id} due to extraction error.")





# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

def removeSpecialChar(text):
    word = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    word =  re.sub(r'\s+', ' ',word).strip()
    return word



def count_sentences(text):
    
    sentences = re.split(r'[.!?]+', text)  # Split based on punctuation
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings
    return sentences

def is_complex(word):
    """Check if a word has 2 or more syllables."""
    clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()  # Remove punctuation & lowercase
    return textstat.syllable_count(clean_word) >= 2 if clean_word else False

def syllabus_word_count(text):

        # Finding all vowels in the text
    vowels = re.findall(r'[AEIOUaeiou]', text)

    # Finding words that end with 'es' or 'ed'
    words_ending_es_ed = re.findall(r'\b\w+(?:es|ed)\b', text)

    return len(vowels),len(words_ending_es_ed)

def PersonalPronouns(text):
    pattern = r'\b(I|me|my|mine|we|us|our|ours|you|your|yours|he|him|his|she|her|hers|it|its|they|them|their|theirs)\b'

    # Finding all occurrences
    personal_pronouns = re.findall(pattern, text, re.IGNORECASE)

    # Count of personal pronouns
    return len(personal_pronouns)



# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

def ExtractArticles():
    text_files = "Extracted_Articles"
    content = []

    # Read all text files from the directory
    for file in os.listdir(text_files):
        with open(os.path.join(text_files, file), "r", encoding="utf-8") as f:
            content.append({"id": file.split(".")[0],"text": f.read()})  # Store text content in the list`

    return content


def clean_text(text_list):
    cleaned_texts = []

    for data in text_list:
        words = removeSpecialChar(data['text'])
        tokens = word_tokenize(words.lower())  # Tokenize each text
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords and punctuation
        cleaned_texts.append({"id":data['id'],'content':filtered_tokens})
    
    return cleaned_texts

text_data = ExtractArticles()

# Process and print cleaned text
cleaned_articles = clean_text(text_data)


metrics = []

def TextAnalysis(text):
    MasterDictionary = "MasterDictionary"

    words = []
    for files in os.listdir(MasterDictionary):
        with open(os.path.join(MasterDictionary, files), "r") as f:
            words.append({"type": files.split(".")[0],"words":f.read().splitlines()})
    if text:
        positive_score = 0
        negative_score = 0
        polarity_score = 0
        subjectivity_score = 0
        Tokenize_word_count = 0

        for cleaned_data in text:
            for punket in words:
                if punket['type'] == "positive-words":
                    positive_score = sum(1 for data in cleaned_data['content'] if data in punket['words'])
                if punket['type'] == "negative-words":
                    negative_score = sum(-1 for data in cleaned_data['content'] if data in punket['words'])
                    negative_score=negative_score*-1
            polarity_score = ((positive_score - negative_score)/(positive_score + negative_score)+0.000001)
            subjectivity_score =((positive_score + negative_score)/(len(cleaned_data["content"]))+0.000001)

            Tokenize_word_count = len(cleaned_data["content"])
            metrics.append({"id": cleaned_data['id'], "Positive Score": positive_score,"Negative Score": negative_score,"Polarity Score": polarity_score,"Subjectivity Score":subjectivity_score,
                            "Word Count":Tokenize_word_count})
                


TextAnalysis(cleaned_articles)
# print(result)

sentence = []

def AnalysisReadabiliy():
    Average_sentence_length = 0
    Percentage_of_complex_Words = 0
    Fog_index = 0
    Average_Number_of_Words_Per_Sentence = 0
    complex_word_count = 0
    Total_number_of_character_per_words = 0
    words = ExtractArticles()
    

    for data in words:
        raw_text = removeSpecialChar(data['text'])
        sentences = count_sentences(data['text'])
        
        split_text = raw_text.split(" ")
        each_character_count = [len(data) for data in split_text]

        word_count = len(raw_text.split())

        sentence_count = len(sentences)  # Count sentences
        
        complex_word = [word for word in raw_text if is_complex(word)]
       
        Percentage_of_complex_Words = word_count / len(complex_word)

        Average_sentence_length = word_count/sentence_count 
        Fog_index = 0.4 * (Average_sentence_length + Percentage_of_complex_Words)
        Average_Number_of_Words_Per_Sentence = word_count / len(raw_text)

        complex_word_count = len(complex_word)
        

        vowels_count, consonent_list = syllabus_word_count(raw_text)

        personal_pronoun_word_count = PersonalPronouns(raw_text)

        Total_number_of_character_per_words = sum(each_character_count) / word_count

        sentence.append({"id":data["id"],"Average Sentence Length":Average_sentence_length ,"Percentage Of Complex Words":Percentage_of_complex_Words,"Fog Index":Fog_index,
                        "Average Number Of Words Per Sentence":Average_Number_of_Words_Per_Sentence,"Complex Word Count":complex_word_count,"Syllabus Count Per Word":vowels_count,"endswith 'ed' 'es' words Count":consonent_list,
                        "Personal Pronouns":personal_pronoun_word_count,"Average Word Length":Total_number_of_character_per_words})

AnalysisReadabiliy()



# Convert list2 into a dictionary for quick lookup
dict2 = {item['id']: item for item in sentence}

# Merge list2 data into list1
for item in metrics:
    if item['id'] in dict2:
        item.update(dict2[item['id']])  # Append data from list2

# Print the merged result

df = pd.DataFrame(metrics)

df.to_excel("Output.xlsx")
