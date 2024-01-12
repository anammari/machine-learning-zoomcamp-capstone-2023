import requests

url = 'http://localhost:9000/2015-03-31/functions/function/invocations'

data = {'url': 'https://i.imgur.com/23MiAxv.jpg'} #large_carnivores (tiger)

result = requests.post(url, json=data).json()
print(result)