import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


def get_transcript_links(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # Wait for the initial load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "vid-materials"))
    )

    # Click the "Show All" button if it exists
    try:
        show_all_button = driver.find_element(By.XPATH, "//button[text()='Show All']")
        show_all_button.click()
        time.sleep(2)  # Wait for content to load
    except:
        print("'Show All' button not found. Proceeding with scrolling.")

    # Scroll to load all episodes
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Additional scrolling to ensure everything is loaded
    body = driver.find_element(By.TAG_NAME, "body")
    for _ in range(10):  # Scroll up and down a few times
        body.send_keys(Keys.END)
        time.sleep(0.5)
        body.send_keys(Keys.HOME)
        time.sleep(0.5)

    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, "html.parser")

    transcript_links = []
    for div in soup.find_all("div", class_="vid-materials"):
        for a in div.find_all("a", href=True):
            if "transcript" in a["href"].lower():
                transcript_links.append(urljoin(url, a["href"]))

    print(f"Total transcript links found: {len(transcript_links)}")
    return transcript_links


def extract_conversation(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    segments = soup.find_all("div", class_="ts-segment")

    conversation = []
    for segment in segments:
        name_span = segment.find("span", class_="ts-name")
        text_span = segment.find("span", class_="ts-text")

        name = name_span.text.strip() if name_span else ""
        text = text_span.text.strip() if text_span else ""

        if text:
            if name:
                conversation.append(f"{name}: {text}")
            else:
                conversation.append(text)

    return "\n\n".join(conversation)


def save_transcript(conversation, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(conversation)


def process_all_transcripts(main_url):
    transcript_links = get_transcript_links(main_url)

    if not os.path.exists("transcripts"):
        os.makedirs("transcripts")

    for i, link in enumerate(transcript_links, 1):
        print(f"Processing transcript {i}/{len(transcript_links)}: {link}")
        try:
            conversation = extract_conversation(link)
            if conversation:
                filename = f"transcripts/transcript_{i}.txt"
                save_transcript(conversation, filename)
                print(f"Saved transcript to {filename}")
            else:
                print(f"No content extracted from {link}")
        except Exception as e:
            print(f"Error processing {link}: {str(e)}")

        time.sleep(1)


# Example usage
main_url = "https://lexfridman.com/podcast/"
process_all_transcripts(main_url)


# import requests
# from bs4 import BeautifulSoup
# import re

# def extract_conversation(url):
#     # Send a GET request to the URL
#     response = requests.get(url)

#     # Check if the request was successful
#     if response.status_code != 200:
#         print(f"Failed to retrieve the page. Status code: {response.status_code}")
#         return None

#     # Parse the HTML content
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Find all ts-segment divs
#     segments = soup.find_all('div', class_='ts-segment')

#     conversation = []
#     for segment in segments:
#         name = segment.find('span', class_='ts-name').text.strip()
#         text = segment.find('span', class_='ts-text').text.strip()

#         # Combine name and text, but only if name is not empty
#         if name:
#             conversation.append(f"{name}: {text}")
#         else:
#             conversation.append(text)

#     # Join the conversation parts
#     full_conversation = "\n\n".join(conversation)

#     return full_conversation

# url = "https://lexfridman.com/jordan-jonas-transcript"
# conversation = extract_conversation(url)

# if conversation:
#     print(conversation)

#     with open('conversation_transcript.txt', 'w', encoding='utf-8') as f:
#         f.write(conversation)
