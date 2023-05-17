# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


import time
import random

# start the web driver
driver = webdriver.Chrome()
# open the website
driver.get("https://gibber.cc/playground/index.html")
# get the html text
html =driver.page_source
action = ActionChains(driver)
driver.find_element(By.CLASS_NAME, "cm-variable").click()
action.key_down(Keys.SHIFT).key_down(Keys.ENTER).key_up(Keys.ENTER).key_up(Keys.SHIFT).perform()
