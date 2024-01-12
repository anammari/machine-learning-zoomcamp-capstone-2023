from lambda_function_local import lambda_handler
import json
event = {'url': 'https://i.imgur.com/23MiAxv.jpg'} #large_carnivores (tiger)
result = lambda_handler(event, None)
json_str = json.dumps(result, indent=4)
print(json_str)