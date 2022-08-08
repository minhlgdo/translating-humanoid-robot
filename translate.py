from lib2to3.pgen2.token import OP
from selenium import webdriver
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options
import httplib

def translate(word):
    translated_word = ""
    options = Options()
    options.add_argument('--headless')
    browser = webdriver.Firefox(options=options)
    website = 'https://translate.google.com/?sl=en&tl=da&text=' + word + '&op=translate'
    
    try:
        browser.get(website)
        browser.implicitly_wait(3)
        # sleep(5)

        # choose the translation
        translated_word = browser.find_element_by_xpath(
            "//span[@class='Q4iAWc']").text
        print("Translate: " + translated_word)
    except httplib.BadStatusLine:
        translated_word = "ERROR"
    browser.quit()
    return translated_word

# main("CONVERSES")
