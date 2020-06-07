import torch
from tqdm import tqdm
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer

MIN_LEN = 55
MAX_LEN = 256
MAX_SEQ_LEN = 1024
DEFAULT_DEVICE = 'cuda'
source_path = Path('~/lds/encoding/test.source')

model = BartForConditionalGeneration.from_pretrained('bart-large').to(DEFAULT_DEVICE)
model.load_state_dict(torch.load(Path('~/lds/ckpt/bart_1024.ckpt'))['state_dict'])
model.eval()

tokenizer = BartTokenizer.from_pretrained('bart-large')
examples = [' ' + line.rstrip() for line in open(source_path).readlines()]

def generate_summary(sample):
    sample_enc = tokenizer.batch_encode_plus([sample], max_length=MAX_SEQ_LEN, return_tensors='pt', pad_to_max_length=True)
    sample_summary_enc = model.generate(
        input_ids = sample_enc['input_ids'].to(DEFAULT_DEVICE),
        attention_mask = sample_enc['attention_mask'].to(DEFAULT_DEVICE),
        num_beams = 4,
        length_penalty = 2.0,
        max_length = MAX_LEN + 2,  # +2 from original because we start at step=1 and stop before max_length
        min_length = MIN_LEN + 1,  # +1 from original because we start at step=1
        no_repeat_ngram_size = 3,
        early_stopping = True,
        decoder_start_token_id = model.config.eos_token_id,
      )
    sample_summary = [tokenizer.decode(idx, skip_special_tokens = True, clean_up_tokenization_spaces = False) for idx in sample_summary_enc]
    return sample_summary[0]

test_sample = examples[0]
test_summary = generate_summary(test_sample)
print('Test')
print(test_summary)

summaries = []
for idx, each in tqdm(enumerate(examples[:10])):
    output = generate_summary(each)
    summaries.append(output) 

with open('./bart_output.txt', 'w') as fout:
    for summary in summaries:
        fout.write(summary + '\n')
        
fout.close()