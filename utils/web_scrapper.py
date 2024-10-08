import os
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from pyvirtualdisplay import Display
import requests
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import logging
import openai
from utils.text_utils import get_or_create_vector_store
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium.webdriver.firefox.service import Service

no_of_pages_serp =  int(os.getenv("NO_OF_SERP_PAGES"))
no_of_results_serp =  int(os.getenv("NO_OF_SERP_RESULTS"))


def scrape_site(url):
    scraped_text = ""
    try:
        print("scraping url", url)
        driver = build_web_driver()
        if driver is not None:
            print("scraping url 1", url)
            driver.get(url)
            print("scraping url 2", url)
            print("scraping url 3", url)
            element_present = EC.presence_of_element_located((By.XPATH, "/html/body"))
            print("scraping url 4", url)
            WebDriverWait(driver, 10).until(element_present)
            html = driver.find_element(By.XPATH, "/html/body").text
            print("scraping url 5", url)
            driver.quit()
            scraped_text = refine_text(html)
        else:
            scraped_text += "NA"
    except TimeoutException:
        print("Timed out waiting for page to load")
        scraped_text += "NA"
    print("scraping completed")
    return scraped_text



def build_web_driver():
    try:
        # Set up WebDriver for Firefox (GeckoDriver)
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--no-sandbox")
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-notifications")

        # Random user-agent to mimic a real browser
        ua = UserAgent()
        userAgent = ua.random
        logging.info(f"Using user-agent: {userAgent}")
        options.set_preference("general.useragent.override", userAgent)

        # Ensure the geckodriver path is correct
        gecko_path = os.getenv("GECKO_DRIVER_PATH")
        if not gecko_path:
            service = Service(GeckoDriverManager().install())
        else:
            service = Service(executable_path=gecko_path)
        # Initialize GeckoDriver with options
        driver = webdriver.Firefox(service=service, options=options)
        return driver
    except Exception as e:
        logging.error(f"Error initializing WebDriver: {e}")
        return None

def serp_scrap_results(query):
    logging.info(f"Scraping for: {query}")

    driver = build_web_driver()
    if driver is None:
        logging.error("WebDriver not initialized")
        return []

    url_array = []
    try:
        # Load Google search page
        url = 'https://www.google.com/'
        driver.get(url)

        # Search for keyword
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)

        for page in range(no_of_pages_serp):
            # Wait for the search results page to load
            try:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '.g'))
                WebDriverWait(driver, 10).until(element_present)
            except TimeoutException:
                logging.warning("Timed out waiting for page to load")
                break

            # Parse the search results
            search_results = driver.find_elements(By.CSS_SELECTOR, '.g')
            search_results = search_results[:no_of_results_serp]
            for result in search_results:
                try:
                    link = result.find_element(By.CSS_SELECTOR, 'a').get_attribute('href').split('#')[0]
                    url_array.append(link)
                except Exception as e:
                    logging.error(f"Error extracting link: {e}")

            # Click on the next page
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, '#pnnext')
                next_button.click()
            except Exception as e:
                logging.warning(f"No more pages or error clicking next: {e}")
                break
    finally:
        driver.quit()

    return url_array




def refine_text(text):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt_context = '''
    We have also extracted following text information from a website:\n
    '''
    prompt_context += text

    prompt = '''
        Your task is to refine the text and provide a clean version of the text and it is within your capabilities.
        Specifically, if there are the following types of conent please remove them:
        1. **Advertisements and Promotional Content**: Any content aimed at selling products, services, or promoting the website itself.
        2. **Navigation Links and Menus**: Links to other sections of the website that do not add to the main content.
        3. **Disclaimers and Legal Notices**: Standard disclaimers or legal information not pertinent to the main content.
        4. **Generic Greetings and Intros**: Standard greetings, intros, or welcome messages that do not contribute to the core information.
        5. **Social Media Links and Share Buttons**: Links to follow on social media or share the content.
        6. **Subscription Prompts**: Requests for readers to subscribe to newsletters or updates.
        7. **Contact Information**: General contact information that does not relate to the main content.
        8. **Boilerplate Text**: Standardized text that is repeated across multiple pages without specific relevance to the current content.

        However, it is crucial to retain all important information related to the main topic.That includes facts, figures, statistics, analysis, and any other relevant data. If not all the types of content mentioned above are present, can be ignore the ones that are not relevant. But do not remove any relevant content in order to follow the above rules.
        Also the most important rule is to ensure that the refined text follows the same structure and flow as the original text. There shouldnt be any change in the meaning of the text.
        Do not try to rewrite or rearrange or paraphrase the text. Just remove the irrelevant content. This is not a creative writing task, just a task to remove irrelevant content.
       '''

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text=prompt_context)
    print("chunks ready")
    vector_store = get_or_create_vector_store(chunks, store_name="vector_store")
    print("vector store ready")
    docs = vector_store.similarity_search(query=prompt, k=3)
    print("docs ready")
    print(docs)
    llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo-instruct")
    print("llm ready")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    print("chain loaded")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=prompt)

    return response

def get_market_cap_by_company_name(company_name):
    google_finance_url = find_google_finance_link(company_name)
    if google_finance_url:
        return get_market_cap(google_finance_url)
    else:
        return f"Google Finance link for {company_name} not found."
        
def find_google_finance_link(company_name):
    query = f"{company_name} googlefinance"
    search_url = f"https://www.google.com/search?q={query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    }
    
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        

        for link in soup.find_all('a', href=True):
            href = link['href']
            if "https://www.google.com/finance/quote/" in href:
                return href.split("&")[0] 
    return None

def get_market_cap(google_finance_url):
    response = requests.get(google_finance_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        elements = soup.find_all('div', class_='gyFHrc')
        
        for element in elements:
            if 'Market cap' in element.text:
                value_div = element.find('div', class_='P6K39c')
                if value_div:
                    return value_div.text.strip()
                else:
                    return "Market cap value not found"
        return "Market cap not found"
    else:
        return f"Failed to retrieve data. Status code: {response.status_code}"



if __name__ == '__main__':
    print(scrape_site("https://www.tipranks.com/stocks/amrx/forecast"))
