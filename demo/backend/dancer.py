class Document:
   def __init__(self, raw_example):
      self.titles = raw_example['section_names']
      self.sections = utils.concate_list(raw_example['sections'])

class Dancer:
   def __init__(self,
                model_name='bart-large',
                max_seq_len=1024,
                ckpt='./model/dancer.ckpt'):
        
      self.max_seq_len = max_seq_len
      self.tokenizer = BartTokenizer.from_pretrained(model_name)
      self.model = BartForConditionalGeneration.from_pretrained(model_name)
      if ckpt is not None:
         self.model.load_state_dict(torch.load(ckpt)['state_dict'])
      self.model.eval()

   def _divide(self, document):
      base_sections = [text for text in document.sections]
      return base_sections
    
   def generate_section_summary(self, section, min_length=10, max_length=50, num_beams=4):
      input_enc = self.tokenizer.batch_encode_plus([section], max_length=self.max_seq_len, return_tensors='pt', pad_to_max_length=True)
      output_enc = self.model.generate(
               input_ids=input_enc['input_ids'],
               attention_mask=input_enc['attention_mask'],
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
