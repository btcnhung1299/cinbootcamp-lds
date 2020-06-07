import utils
from rouge import Rouge
from tqdm import tqdm

class Article:
    def __init__(self, raw_example):
        self.index = raw_example['article_id']
        self.titles = raw_example['section_names']
        self.sections = utils.concate_list(raw_example['sections'])
        tokenized_summary = utils.tokenize_list(raw_example['abstract_text'], remove_keyword=True)
        self.summary = ' '.join(utils.concate_list(tokenized_summary)).lower()

    def get_primary_indices(self):
        primary_indices, others = [], []
        for idx, title in enumerate(self.titles):
            if utils.is_primary_section(title):
                primary_indices.append(idx)
            else:
                others.append(idx)
        return primary_indices, others

import json
article = None
scorer = Rouge()

def update_rouge(index, title: str, sections: str, target: str):
    global scorer
    if not sections:
        r1 = r2 = rl = 0
    else:
      try:
        scores = scorer.get_scores(sections.lower(), target)[0]
      except:
        return ''
      r1 = scores['rouge-1']['p']
      r2 = scores['rouge-2']['p']
      rl = scores['rouge-l']['p']
    return '{},{},{},{},{}\n'.format(index, title, r1, r2, rl)
    
primary_write = open('primary_rouge.csv', 'w')
other_write = open('other_rouge.csv', 'w')

for line in tqdm(open('./encoding/train.txt', 'r')):
    data = json.loads(line)
    article = Article(data)
    idx = article.index
    primary_indices, other_indices = article.get_primary_indices()
    target = article.summary

    for each in primary_indices:
      s = update_rouge(idx, article.titles[each], article.sections[each], target)
      primary_write.write(s)
    for each in other_indices:
      s = update_rouge(idx, article.titles[each], article.sections[each], target)
      other_write.write(s)
    break

primary_write.close()
other_write.close()