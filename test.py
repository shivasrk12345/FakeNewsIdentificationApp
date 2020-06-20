#from .chrome.webdriver import WebDriver as Chrome  # noqa

from selenium import webdriver
driver = webdriver.Chrome("C:/Users/gant0006/Downloads/chromedriver.exe")
driver.get("https://www.youtube.com/")

#C:\Program Files (x86)\Google\Chrome\Application