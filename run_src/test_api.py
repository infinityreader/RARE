# from openai import OpenAI
# client = OpenAI(base_url="http://172.16.34.22:6999/v1", api_key="original")
# sentence = "What's asthma?"
# messages = [{"role": "system", "content": "You are a helpful chatbot, please chat with me."},
#             {"role": "user", "content": sentence}]
# completion = client.chat.completions.create(
#   model="llama31_8b_with_openai_api",
#   messages=messages)
# print(completion.choices[0].message)
import requests
import json
# num_searches = 3
# search_query = "lung cancer"
# query = "http://172.16.34.1:5000/api/search?query="+search_query+"&k="+str(num_searches)
# x = requests.get(query)
# jsobj = json.loads(x.text)
# import ipdb; ipdb.set_trace()
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro1_s3_r3_textstat_t5_a1_a3_a4_a6_fm.json", "r") as f:
#     data1 = json.load(f)
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro1_s3_r3_textstat_t5_a1_a3_a4_a7_fm.json", "r") as f:
#     data2 = json.load(f)
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_s3_r3_textstat_t5_a1_a3_a4_a6_fm.json", "r") as f:
#     data3 = json.load(f)
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_s3_r3_textstat_t5_a1_a3_a4_a7_fm.json", "r") as f:
#     data4 = json.load(f)
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_a1_a3_a4_fm.json", "r") as f:
#     data0 = json.load(f)

file_name = "/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_SIQA_ro1_chat_a1_a3_a4_fm.json"
with open(file_name, "r") as f:
    data = json.load(f)
print(file_name)
# import ipdb; ipdb.set_trace()
print(data[-1]["num_tested"], data[-1]["accuracy"], data[-1]["num_gen_correct"]/data[-1]["num_tested"], data[-1]["num_correct"], data[-1]["num_gen_correct"])
# print(data[-1]["num_tested"], data[-1]["accuracy"], data[-1]["num_gen_correct"]/data[-1]["num_tested"], data[-1]["num_correct"], data[-1]["num_gen_correct"])
# full_data = data[:-1]
# with open(file_name, "w") as f:
#     json.dump(full_data, f)
# import ipdb; ipdb.set_trace()
# with open("save/example.json", "w") as f:
#     json.dump(data[19], f)
# import ipdb; ipdb.set_trace()
# for idx in range(len(data)):
#     if idx == 249:
#         import ipdb; ipdb.set_trace()
import random

# List of numbers provided
# numbers = [9, 10, 11, 14, 20, 21, 23, 24]
# numbers = [ 7, 9, 17, 22, 24, 31, 35, 37, 38, 39]
# numbers = [4, 6, 9, 10, 13, 18, 20, 26, 32, 43]
# # Calculate 40% of the list size
# num_to_pick = int(len(numbers) * 0.4)
#
# # Pick random numbers
# random_numbers = random.sample(numbers, num_to_pick)
#
# print(random_numbers)



