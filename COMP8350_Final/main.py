# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import time
import random

# start the web driver
driver = webdriver.Chrome()
# open the website
driver.get("https://gibber.cc/playground/index.html")
# get the html text
html =driver.page_source
driver.execute_script("document.getElementsByClassName('cm-variable')[0].innerHTML='f3ffff'")
