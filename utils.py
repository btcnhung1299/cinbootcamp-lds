import json
import string

with open('keyword_section.json', 'r') as fin:
    keyword_section = json.load(fin)

with open('primary_section.json', 'r') as fin:
    primary_section = json.load(fin)


def is_primary_section(title: str):
    """Check whether a section is informative based on its title
    
    Args:
        title: string of characters in section title
        
    Returns:
        True if a section is informative, otherwise False
    """
    
    global primary_section
    
    lower_title = title.lower()
    for keywords_section in primary_section.values():
        for keyword in keywords_section:
            if lower_title.find(keyword) != -1:
                return True
    return False


def remove_title(s: str, threshold=3):
    """Remove section title in improperly tokenized string
    Section title pattern: ".*<section_title>:"
    Ex: "A is apart of B.Methods: Do A before B."
    
    Args:
        s: string of characters
        threshold: the minimum number of words in a section title
        
    Returns:
        New string whose section title removed
        Ex: "A is apart of B. Do A before B."
    
    """
    start_title_idx = 0
    end_title_idx = s.find(':')
    string_changed = False
    
    while end_title_idx != -1:
        # Find the last occurence of '.' before the newly found ':'
        # If there is none of them, return the beginning index of the string
        start_title_idx = max(0, s.rfind('.', start_title_idx, end_title_idx))
        title = s[start_title_idx : end_title_idx + 1] 
        
        deleted = False
        
        # Only consider removing section title if number of words in it (after removed '.' and ':') <= threshold
        if len(title.strip('.:').split()) <= threshold:
            for section in keyword_section.values():
                if deleted:
                    break
                for keyword in section:
                    if title.find(keyword) != -1:
                        replace_str = ' . ' if start_title_idx != 0 else ''
                        s = s.replace(title, replace_str)
                        deleted = True
                        string_changed = True
                        break
        
        if not deleted:
            start_title_idx = end_title_idx + 1
        end_title_idx = s.find(':', start_title_idx)
        
    return string_changed, s


def concate_list(token_list: list):
    return [' '.join(tokens) for tokens in token_list]


def tokenize_list(raw_list: list, remove_keyword=False, remove_sep_tokens=True):
    """Convert a list of sentences to a list of tokenized sentences
    
    Agrs:
        raw_list: List of sentences
        remove_keyword: Whether title should be removed from the sentence
        remove_sep_tokens: Whether seperators should be removed from the sentence
        
    Returns:
        List of tokenized sentences
    """
    
    def _resplit(raw_string: str):
        splitted, new_string = remove_title(raw_string)
        return [raw_string] if not splitted else new_string.split(' . ')
    
    def _normalize(raw_string: str):
        sep_tokens = ['<S>', '</S>']
        res = raw_string
        for token in sep_tokens:
            res = res.replace(token, '')
        return res
        
    token_list = []
    
    for raw_sentence in raw_list:
        raw_sentence = raw_sentence if not remove_sep_tokens else _normalize(raw_sentence)
        sentences = [raw_sentence] if not remove_keyword else _resplit(raw_sentence)
        
        for s in sentences:    
            tokens = s.split()
            if not tokens:
                break
            
            # Since string are splitted by '.', the last character are not reserved
            start, end = 0, len(tokens) - 1
            if tokens[end] not in string.punctuation:
                tokens.append('.')
                end += 1
            token_list.append(tokens[start : end + 1])
    
    return token_list


def compute_lcs(source: list, target: list):
    """Compute the longest common sequence (LCS) between source and target list of words
    
    Args:
        source: List of source tokens.
        target: List of target tokens.
        
    Returns:
        Length of LCS
    """
    
    num_rows, num_cols = len(source), len(target)
    dp = [[0] * (num_cols + 1) for _ in range(num_rows + 1)]
    
    for i in range(1, num_rows + 1):
        for j in range(1, num_cols + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[num_rows][num_cols]


def compute_rouge_l(source: list, target: list, metric="precision"):
    """Compute R-L score between source and target list of words
    
    Args:
        source: List of source tokens.
        target: List of target tokens.
        metric: precision/recall/f1
        
    Returns:
        A single score of specified metric.
    
    """
    len_lcs = compute_lcs(source, target)
    if len_lcs == 0:
        precision = recall = f1 = 0.0
    else:
        precision = len_lcs / len(source)
        recall = len_lcs / len(target)
        f1 = 2 * precision * recall / (precision + recall)
 
    results = {"precision": precision, "recall": recall, "f1": f1}
    return results[metric]