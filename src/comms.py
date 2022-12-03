import requests
from icecream import ic


address = "http://192.168.137.18"
payload = {"rcw": "1000"}

r = requests.get(address, params=payload)
ic(r.text)