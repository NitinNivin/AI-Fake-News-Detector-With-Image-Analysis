import google.generativeai as genai
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time
from urllib.parse import urlparse
from collections import Counter
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from newspaper import Article
import newspaper
import nltk
import numpy as np

# Download nltk resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Load environment variables from .env file
load_dotenv()

# Set the API key from the environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API key is not set. Please define GOOGLE_API_KEY in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Set up the Gemini Pro model
model = genai.GenerativeModel('gemini-pro')

# Image Classification Model Setup
image_model_path = 'best_ai_detector.pth'  # Update this path
image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_ftrs, 2)
image_model.load_state_dict(torch.load(image_model_path, map_location=torch.device('cpu')), strict=False)
image_model.eval()

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    img_tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = image_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return "AI-generated" if predicted.item() == 0 else "Real"

def perform_google_search(query):
    search_url = f"https://www.google.com/search?q={query}&num=10"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()  # Check for bad status code
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        for result in soup.find_all('div', class_='g'):  # Google search result container
            link_tag = result.find('a')
            if link_tag and 'href' in link_tag.attrs:
                link = link_tag['href']
                if link.startswith('http'):
                    links.append(link)

        return links
    except requests.exceptions.RequestException as e:
        print(f"Error during Google search: {e}")
        return []

def extract_text_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)  # added timeout for slow websites
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        text_parts = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])  # Extracting main content text
        text = ' '.join(part.get_text(strip=True) for part in text_parts)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_website_credibility(url):
    reliable_sources = {
        "bbc.com": 2,
        "nytimes.com": 2,
        "reuters.com": 2,
        "apnews.com": 2,
        "who.int": 2,
        "cnn.com": 1.5,
        "ndtv.com": 1.5
    }
    unreliable_sources = {
        "infowars.com": -2,
        "naturalnews.com": -2,
        "beforeitsnews.com": -2
    }

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]  # Remove "www." if present
        if domain in reliable_sources:
            return reliable_sources[domain]
        elif domain in unreliable_sources:
            return unreliable_sources[domain]
        return 0  # Default is neutral
    except Exception as e:
        print(f"Error getting domain: {e}")
        return 0

def detect_suspiciousness(text, target):
    """Detects the stance of a given text towards a target topic and flags suspiciousness.

    Args:
        text: The text to analyze.
        target: The topic or entity to detect stance towards.

    Returns:
        A dictionary containing:
         - stance : The stance (e.g., "supporting", "opposing", "neutral", "unclear", or error)
         - strength : A string describing the strength of the stance ("strong", "weak", "not applicable")
         - is_suspicious: A boolean value indicating if the text is suspicious.
    """
    prompt = f"""
    Analyze the following text and determine the author's stance towards the topic of "{target}".

    Text:
    {text}

    Consider these possible stances:
    - supporting: The author's viewpoint is in favor of "{target}".
    - opposing: The author's viewpoint is against "{target}".
    - neutral: The author expresses no clear opinion or does not have any stance on "{target}".
    - unclear: The author's viewpoint is not clear in the text provided.

    Also provide a judgement of the strength of the authors stance in each category using "strong", or "weak".  If the stance is neutral or unclear, state "not applicable"

    Provide a response that is just in the following format, one value per line with no additional text:
    stance
    strength
    """
    try:
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        stance = lines[0].lower()
        strength = lines[1].lower()

        is_suspicious = False

        # Simple logic for flagging suspiciousness - this can be improved upon
        if strength == "strong":
            if stance == "supporting" or stance == "opposing":
                is_suspicious = True

        return {
            "stance": stance,
            "strength": strength,
            "is_suspicious": is_suspicious,
        }

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return {
            "stance": "error",
            "strength": "not applicable",
            "is_suspicious": False,
        }

def fetch_article_data(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.images
    except Exception as e:
        st.error(f"An error occurred while fetching the article: {e}")
        return None, []

def analyze_query(query):
    links = perform_google_search(query)
    results = []
    stances = []
    url_analysis = {}
    real_score = 0
    fake_score = 0
    suspicious_count = 0

    for link in links:
        text = extract_text_from_url(link)
        if text:
            credibility = get_website_credibility(link)
            time.sleep(1)  # added rate limit wait.
            stance_result = detect_suspiciousness(text, query)
            stances.append(stance_result['stance'])

            score = 0
             #Real Score Calculation
            if stance_result['stance'] in ["supporting", "neutral"] or stance_result['stance'] == 'agree':
              if credibility > 0:
                score += credibility
              else:
                  score += 1
            #Fake Score Calculation
            elif stance_result['stance'] in ["opposing", "unclear"] or stance_result['stance'] == "disagree":
              if credibility < 0:
                  score += credibility
              else:
                  score -= 1

            if stance_result['strength'] == "strong":
                score += 1
            elif stance_result['strength'] == "weak":
               score += 0.5

            if stance_result['is_suspicious']:
                score -= 0.5
                suspicious_count += 1
            else:
              score +=0.5

            if score > 0:
                real_score += score
            elif score < 0:
                fake_score += score

            url_analysis[link] = {"stance": stance_result['stance'], "score": score, "suspicious": stance_result['is_suspicious'] }

            results.append({
                "url": link,
                "text": text,
                "credibility": credibility,
                **stance_result,
                "score": score
            })
    most_common_stance = Counter(stances).most_common(1)[0][0] if stances else "unknown"

    if suspicious_count > len(results)/2:
       if most_common_stance in ["opposing", "unclear"]:
           return results, "Fake", url_analysis, real_score, fake_score
       else:
           return results, "Real", url_analysis, real_score, fake_score
    else:
         return results, most_common_stance, url_analysis, real_score, fake_score



def analyze_headline_stance(headline, url_scores):
    links = perform_google_search(headline)
    real_score = 0
    fake_score = 0
    real_urls = []
    fake_urls = []
    url_analysis = {}


    for link in links:
        text = extract_text_from_url(link)
        if text:
            credibility = get_website_credibility(link)
            time.sleep(1)
            stance_result = detect_suspiciousness(text, headline)
            score = 0

            #Real Score Calculation
            if stance_result['stance'] in ["supporting", "neutral"] or stance_result['stance'] == 'agree':
              if credibility > 0:
                score += credibility
              else:
                  score += 1
            #Fake Score Calculation
            elif stance_result['stance'] in ["opposing", "unclear"] or stance_result['stance'] == "disagree":
              if credibility < 0:
                  score += credibility
              else:
                  score -= 1

            if stance_result['strength'] == "strong":
                score += 1
            elif stance_result['strength'] == "weak":
               score += 0.5

            if stance_result['is_suspicious']:
                score -= 0.5
            else:
              score +=0.5
            if link in url_scores:
                 score += url_scores[link]
            else:
                url_scores[link] = 0
            if score > 0:
                 real_score += score
                 real_urls.append(link)
            elif score < 0:
                fake_score += score
                fake_urls.append(link)
            url_analysis[link] = {"stance": stance_result['stance'], "score": score}



    if real_score > abs(fake_score):
        return "Real", real_urls, url_analysis, url_scores, real_score, fake_score
    else:
        return "Fake", fake_urls, url_analysis, url_scores, real_score, fake_score

def is_blank_image(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3:  # Check if it's an RGB image
        if np.all(image_np == 0) or np.all(image_np == 255) or np.all(image_np == [255, 255, 255]):
           return True
    return False

# Streamlit app
def main():
    st.title("AI Fake News Classifier & Image Detection")

    option = st.radio("Select the functionality you want to access:", ("Ai image", "Truth finder", "URL Checker"))

    if option == "Ai image":
        st.subheader("Image AI Detection")
        uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction = predict_image(image)
            st.write(f"The image is: **{prediction}**")
        else:
            st.write("Please upload an image to classify.")

    elif option == "Truth finder":
        st.subheader("The Truth Finder: Uncovering Facts in the Digital Age")
        user_query = st.text_input("Enter your query:")
        if st.button("Analyze"):
            if user_query:
                results, majority_stance, url_analysis, real_score, fake_score = analyze_query(user_query)
                st.subheader("Stance Analysis Results")
                # Determine overall assessment based on scores
                if real_score > abs(fake_score):
                   overall_assessment = "Real"
                else:
                    overall_assessment = "Fake"

                st.write(f"**Overall Assessment: {overall_assessment}**")
                st.write(f"**Real Score: {real_score}**")
                st.write(f"**Fake Score: {fake_score}**")
                st.write("Supporting Proof:")
                if results:
                    for res in results:
                        st.write(f"- {res['url']} | Stance: {res['stance']}, Score: {res['score']}")
                else:
                    st.write("No articles found.")
            else:
                st.write("Please enter a query.")

    elif option == "URL Checker":
        st.subheader("URL Analysis: Stance Detection and Image Classification")
        url = st.text_input("Enter a URL:")
        if 'url_scores' not in st.session_state:
            st.session_state['url_scores'] = {}  # initialize url_scores if not already created
        url_scores = st.session_state['url_scores']

        if st.button("Analyze URL"):
            if url:
                headline, image_urls = fetch_article_data(url)
                if headline:
                    st.success("Article data fetched successfully!")
                    st.subheader("Headline Analysis:")
                    assessment, supporting_urls, url_analysis, url_scores, real_score, fake_score = analyze_headline_stance(headline, url_scores)
                    st.write(f"**Overall Assessment: {assessment}**")
                    st.write(f"**Real Score: {real_score}**")
                    st.write(f"**Fake Score: {fake_score}**")
                    if assessment == "Real":
                        st.write("Supporting Proof:")
                        for link in supporting_urls:
                            st.write(f"- {link} | Stance: {url_analysis[link]['stance']}, Score: {url_analysis[link]['score']}")
                        if url in url_scores:
                            url_scores[url] += 0.1
                        else:
                            url_scores[url] = 0.1
                    elif assessment == "Fake":
                       st.write("Supporting Proof:")
                       for link in supporting_urls:
                            st.write(f"- {link} | Stance: {url_analysis[link]['stance']}, Score: {url_analysis[link]['score']}")
                       if url in url_scores:
                            url_scores[url] -= 0.1
                       else:
                            url_scores[url] = -0.1


                    if image_urls:
                        st.subheader("Article Image Analysis")
                        for image_url in image_urls:
                            if image_url.startswith(('http://', 'https://')):
                                try:
                                    image_response = requests.get(image_url, timeout=5)
                                    if image_response.status_code == 200:
                                        image = Image.open(BytesIO(image_response.content)).convert('RGB')
                                        if not is_blank_image(image):
                                            image_class = predict_image(image)
                                            st.image(image, caption=f"**Classified as:** {image_class}", use_column_width=True)
                                        else:
                                            st.write("Skipping blank image")
                                    else:
                                        st.write("Failed to load image from the URL.")
                                except Exception as e:
                                    st.write(f"Could not process image: {e}")
                            else:
                                st.write("Invalid image URL; skipping.")
                    else:
                        st.write("No main images found in the article.")
                else:
                    st.warning("Failed to extract article data from URL.")
            else:
                st.warning("Please enter a valid URL.")


if __name__ == "__main__":
    main()