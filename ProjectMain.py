import pandas as pd
import re
#Data cleaning

data = pd.read_csv('houses_parser_initial.csv')

#From the dataframe above, the city column was splitted to two - city and district.
data[['city','district']] = data.city.str.split(",",expand=True,)

#to find area of each appartment, we take the second item from string splitted by commas
data['area'] = data['name'].str.rsplit(',').str[1]

#number of rooms is the first character in the name column
data['roomsNumber'] = data['name'].str[0]

#converting rooms number to int
data['roomsNumber'] =pd.to_numeric(data['roomsNumber'])

#converting price to int
data['price'] = data['price'].str.extract(r'(\d+)')
data['price'] = data['price'].str.replace('[^0-9]', '', regex=True)
# Remove non-numeric characters
data['price'] = pd.to_numeric(data['price'], errors='coerce')

#converting years number to int
data['year'] = pd.to_numeric(data['year'], errors='coerce')

#deleting whitespaces and units, converting to float
data['area'] = data['area'].str.replace(' ', '')
data['area'] = data['area'].str.replace('м²', '')
data['area'] =pd.to_numeric(data['area'])


data.to_csv('houses_parser.csv', index=False)

print(data.head())