import requests

url = 'https://XXX.execute-api.us-east-1.amazonaws.com/test/predict'

data = {'url': 'https://i.imgur.com/23MiAxv.jpg'} #large_carnivores (tiger)

result = requests.post(url, json=data).json()
print(result)