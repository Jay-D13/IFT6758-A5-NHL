import pandas as pd
import json

df = pd.read_pickle('./ift6758/features/TrainValSets.pkl')
json_data_test = df.iloc[3:5].to_json()

#json_data = json.loads('{"empty_net":{"3":0,"4":0},"is_goal":{"3":0,"4":0},"distance":{"3":58.9406481132,"4":62.60990337},"angle":{"3":-14.7435628365,"4":26.5650511771}}')
json_data = json.loads('{"empty_net":0.0,"is_goal":0.0,"distance":58.9406481132,"angle":-14.7435628365}')
#print(json_data)
list_keys = list(json_data.keys())
print(list_keys[0])
if isinstance(json_data[list(json_data.keys())[0]], float):
    df_json = pd.DataFrame(json_data, index=[0])
else:
    df_json = pd.DataFrame(json_data)

print(df_json)