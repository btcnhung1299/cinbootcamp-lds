import torch
import json
import utils
from transformers import BartTokenizer, BartForConditionalGeneration

class Document:
   def __init__(self, raw_example, from_dataset=True):
      self.titles = raw_example['section_names']
      if from_dataset:
         self.sections = utils.concate_list(raw_example['sections'])
      else:
         self.sections = raw_example['sections']

class Dancer:
   def __init__(self,
                model_name='facebook/bart-large',
                max_seq_len=1024,
                model_path='./model/dancer.pt',
                device='cuda'):
      self.max_seq_len = max_seq_len
      self.device = device
      self.tokenizer = BartTokenizer.from_pretrained(model_name)
      if model_path is not None:
        self.model = BartForConditionalGeneration.from_pretrained(model_name, state_dict=torch.load(model_path))
      else:
        self.model = BartForConditionalGeneration.from_pretrained(model_name)  
      
      self.model.to(device)      
      self.model.eval()

   def _divide(self, document, main_sections=False):
      if main_sections:
        base_sections = [text for title, text in zip(document.titles, document.sections) if utils.is_primary_section(title)]          
      else:
        base_sections = [text for text in document.sections]
      return base_sections
    
   def generate_section_summary(self, section, min_length=10, max_length=50, num_beams=4):
      input_enc = self.tokenizer.batch_encode_plus([section], max_length=self.max_seq_len, return_tensors='pt', pad_to_max_length=True)
      output_enc = self.model.generate(
               input_ids=input_enc['input_ids'].to(self.device),
               attention_mask=input_enc['attention_mask'].to(self.device),
               nem_beams=num_beams,
               max_length=max_length + 2,
               min_length=min_length + 1,
               no_repeat_ngram_size=3,
               early_stopping=True,
               decoder_start_token_id=self.model.config.eos_token_id
               )
      output = ' '.join(self.tokenizer.decode(idx, skip_special_tokens=True, clean_up_tokenization_spaces=False) for idx in output_enc)
      end_idx = output.rfind(' .')
      if end_idx != -1:
         output = output[:end_idx + 2]
      return output

   def generate_document_summary(self, document: Document):
      base_sections = self._divide(document)
      partial_summaries = [self.generate_section_summary(e) for e in base_sections]
      # final_summary = ' '.join(partial_summaries)
      return '; \n'.join(partial_summaries)