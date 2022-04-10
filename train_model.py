from transformers import (
AutoModelWithLMHead,
AutoConfig,
Trainer,
AutoTokenizer,
TextDataset,
DataCollatorForLanguageModeling,
TrainingArguments)

import argparse
from utils import get_data_from_file

def train_model(text_path, output_file, epochs, model_name, tokenizer_name, batch_size, cache_dir = "cache"):

    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model.parallelize()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(
      tokenizer=tokenizer,
      file_path=text_path,
      block_size=256
    )
    
    training_args =TrainingArguments(
    output_dir=output_file,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=100,
    save_steps=2000,
    logging_steps=10,
    prediction_loss_only=True
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()


def construct_training_file(texts: list, labels:list, training_file_path: str, prompt_text: str, prompt_label: str, verbalizers: dict(), causal: bool):

    with open(training_file_path, 'w') as f:
        for text, label in zip(texts, labels):
            if causal:
                f.write(f'<BOS> {prompt_label}: {verbalizers[label]}\n')
                f.write(f'{prompt_text}: {text} <EOS>\n\n')
                
            else:
                f.write(f'<BOS> {prompt_text}: {text}\n')
                f.write(f'{prompt_label}: {verbalizers[label]} <EOS>\n\n')


def run(filepath, training_file_path, prompt_text, prompt_label, verbalizers, causal, model_output_file, epochs, model_name, tokenizer_name, batch_size):
    training_samples, labels = get_data_from_file(filepath)
    construct_training_file(training_samples, labels, training_file_path, prompt_text, prompt_label, verbalizers, causal)
    train_model(text_path=training_file_path, output_file=model_output_file, epochs=epochs, model_name=model_name, tokenizer_name=tokenizer_name, batch_size=batch_size)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', type=str)
    parser.add_argument('--training-file-path', type=str)
    parser.add_argument('--prompt-text', type=str)
    parser.add_argument('--prompt-label', type=str)
    parser.add_argument('--verbalizer-0', type=str)
    parser.add_argument('--verbalizer-1', type=str)
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--model-output-file', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--tokenizer-name', type=str)
    parser.add_argument('--batch-size', type=int)

    args = parser.parse_args()

    run(filepath=args.filepath, 
        training_file_path=args.training_file_path, 
        prompt_text=args.prompt_text, 
        prompt_label=args.prompt_label, 
        verbalizers={0: args.verbalizer_0, 1: args.verbalizer_1}, 
        causal=args.causal,
        model_output_file=args.model_output_file, 
        epochs=args.epochs, 
        model_name=args.model_name, 
        tokenizer_name=args.tokenizer_name, 
        batch_size=args.batch_size
        )

        
            

