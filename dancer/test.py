import json

with open('primary_section.json', 'r') as fin:
    primary_section = json.load(fin)
    
for each in primary_section.keys():
    primary_section[each] = set(primary_section[each])