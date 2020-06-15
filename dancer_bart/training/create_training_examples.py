import utils

def create_partial_tokens(section_titles: list, sections: dict, ref_summary: list):
    """Create partial target tokens by mapping tokens in each reference summary sentence 
    to most appropriate base sections based on ROUGE-L precision
    
    Args:
        section_titles: List of section titles.
        sections: Dictionary of (title, body) where body is list of tokenized sentences in the body.
        ref_summary: List of tokenized sentences in the reference summary.
    
    Returns:
        Dictionary of (title, partial_target_summary) where partial_target_summary is list of tokens in the corresponding section.
    """
    
    target_tokens = {title: [] for title in section_titles}
 
    for ref_sentence_tokens in ref_summary:
        max_rouge_l = 0
        target_sections = set()
        
        # Find section(s) whose the most "similar" sentence to `ref_sentence_tokens`
        for title in section_titles:
            for section_sentence_tokens in sections[title]:
                rouge_l = utils.compute_rouge_l(section_sentence_tokens, ref_sentence_tokens)
                if rouge_l > max_rouge_l:
                    max_rouge_l = rouge_l
                    target_sections = set()
                    target_sections.add(title)
                elif rouge_l == max_rouge_l:
                    target_sections.add(title)
        
        # Add `ref_sentence_tokens` to the appropriate section(s)
        for section in target_sections:
            target_tokens[section].append(ref_sentence_tokens)

    return target_tokens

def create_training_example(raw_example: dict):
    """
    Create training examples of principal sections
    
    Args:
        raw_example: Dictionary of sections in an article
        
    Returns:
        List of titles
        Dictionary of sections
        Dictionary of partial target summaries
    """
    
    titles = [title for title in raw_example['section_names'] if utils.is_primary_section(title)]
    sections = {title: body for title, body in zip(titles, raw_example['sections'])}
    
    tokenized_sections = {title: utils.tokenize_list(body, remove_sep_tokens=False) for title, body in sections.items()} 
    tokenized_target = utils.tokenize_list(raw_example['abstract_text'], remove_keyword=True)
    #print(*tokenized_target, sep='-\n')
    
    partial_target_tokens = create_partial_tokens(titles, tokenized_sections, tokenized_target)
    partial_target = {title: utils.concate_list(token_list) for title, token_list in partial_target_tokens.items()}
   
    return titles, sections, partial_target

def create_sample(raw_example: dict):
    titles = raw_example['section_names']
    sections = utils.concate_list(raw_example['sections'])
    blank_sections = [idx for idx in range(len(titles)) if not sections[idx].strip()]
    offset = 0
    for idx in blank_sections:
        titles.pop(idx - offset)
        sections.pop(idx - offset)
        offset += 1
    
    assert len(titles) == len(sections)
    return {'section_names': titles, 'sections': sections}

# ==================== MAIN =========================


import json
source_examples = []
target_examples = []
sample_path = open('samples.txt', 'w')

for article in open('val.txt', 'r'):
    article = article.strip()
    if not article:
        continue
    data = json.loads(article)
    sample = create_sample(data)
    sample_json = json.dumps(sample)
    sample_path.write(sample_json + '\n')

"""
n = 0
for article in open('val.txt', 'r'):
    article = article.strip()
    if not article:
        continue
    
    data = json.loads(article)
    titles, X, y = create_training_example(data)
    
    for i in titles:
        if not X[i] or not y[i]:
            continue
        source_examples.append(X[i])
        target_examples.append(y[i])
    n += 1
    if n == 30:
        break
"""
