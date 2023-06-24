from selenium import webdriver
#chrome browser is used
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from time import sleep
import pandas as pd
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import Select


#list for announcement links
links = []
#list for each of 7 pages with announcements
pages = []
#instance of webdriver
driver = webdriver.Chrome(executable_path = r'C:\Users\User\Downloads\chromedriver')
driver.maximize_window()
driver.get('https://krisha.kz')
# clicking to drop down for regions
# region_searh = driver.find_element_by_id('region-selected-value').click();
region_searh = driver.find_element("id", "region-selected-value").click();
# waiting
sleep(3);
# choosing Almaty
dropCity = Select(
    driver.find_element("xpath", value="//div[@class='element-region-dropdown-selects']/div/select")).select_by_value(
    "2");

sleep(3)
# clicking choose button
select_button = driver.find_element("xpath",
                                    value="//div[@class='leveled-select is-visible']/a[@class='btn btn-primary region-dropdown-action region-dropdown-action-apply']").click()

sleep(3)
# clicking search button

search_button = driver.find_element("xpath",
                                    value="//div[@class='search-block-submit']/button[@class='kr-btn kr-btn--blue kr-btn--medium search-btn-main']").click()
sleep(3)
# retrieving all 7 page links
for i in range(1, 8):
    page = driver.current_url + '?page=' + str(i);
    pages.append(page)
# in page links gathering announcement links
for p in pages:
    driver.get(p)
    ne_page_links = driver.find_elements("xpath", "//section[@class='a-list a-search-list a-list-with-favs']")
    for section in ne_page_links:
        houses = section.find_elements("xpath", "//div[@class='a-card__inc']/a[@class='a-card__image  ']")
        for link in houses:
            links.append(link.get_attribute("href"))

driver.close()

print(len(links))

#function to clean data(price)
def clean(text, symbols=['\n']):
    for symbol in symbols:
        text = text.replace(symbol, '')
    return text.strip()


# importing exception module for try,except block
from selenium.common.exceptions import NoSuchElementException

# list for each announcement info
infos = []
# retrieving name, phone, city, price, author for each link
for l in links:
    driver1 =  webdriver.Chrome(executable_path = r'C:\Users\User\Downloads\chromedriver')
    driver1.maximize_window()
    driver1.get(l)

    name = driver1.find_element("xpath", value="//div[@class='offer__advert-title']/h1").text
    city = driver1.find_element("xpath", value="//div[@class='offer__location offer__advert-short-info']/span").text
    price = driver1.find_element("xpath", value="//div[@class='offer__sidebar-header']").text
    elements = driver1.find_elements("xpath", value="//div[@class='offer__info-item']")
    for element in elements:
        if "Год постройки" in element.text:
            year = element.text.split("\n")[1]

    # cleaning price
    price = clean(price, ['\n', 'от', '〒', ' '])
    # price = pd.to_numeric(price)

    info = {'name': name, 'price': price, 'city': city, 'year': year}
    infos.append(info)
    driver1.close()

data = pd.DataFrame(infos)
data.head()
data.to_csv('houses_parser_initial.csv', index=False)