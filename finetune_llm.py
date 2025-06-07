import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024
    )

def main():
    # Hyperparameters
    model_name = "meta-llama/Llama-2-7b-chat-hf"  
    output_dir = "./qlora-graph-model"
    micro_batch_size = 4
    gradient_accumulation_steps = 4
    epochs = 3
    lr = 1e-4

    train_desc_path = "./data/train/description"
    train_graph_path = "./data/train/graph"
    valid_desc_path = "./data/valid/description"
    valid_graph_path = "./data/valid/graph"

    from datasets import Dataset, DatasetDict
    import glob

    def load_texts_with_prompt(desc_path, graph_path):
        desc_files = sorted(glob.glob(os.path.join(desc_path, "*")))
        graph_files = sorted(glob.glob(os.path.join(graph_path, "*")))
        samples = []
        for d_file, g_file in zip(desc_files, graph_files):
            with open(d_file, "r") as fdesc:
                desc_text = fdesc.read().strip()
            with open(g_file, "r") as fgraph:
                graph_text = fgraph.read().strip()
            prompt = f"""You are an advanced Graph Generation AI. 
Read the following description carefully, then produce the final edge list (or adjacency list) that matches all the provided properties.

DESCRIPTION:
{desc_text}

OUTPUT:
{graph_text}
"""
            samples.append({"text": prompt})
        return samples

    train_data = load_texts_with_prompt(train_desc_path, train_graph_path)
    valid_data = load_texts_with_prompt(valid_desc_path, valid_graph_path)
    ds_train = Dataset.from_list(train_data)
    ds_valid = Dataset.from_list(valid_data)
    dataset = DatasetDict({"train": ds_train, "valid": ds_valid})

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    def tokenize_fn(ex):
        return tokenize_function(ex, tokenizer)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        learning_rate=lr,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(output_dir)

    # Inference demo
    text_prompt = """You are an advanced Graph Generation AI.
We have the following description: 
Graph with 45 nodes and 44 edges, no triangles, 7 communities.

Now produce the final edge list:"""
    inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
    print("Generated:\n", tokenizer.decode(gen_output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
