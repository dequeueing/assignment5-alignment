import torch
import torch.nn.functional as F

from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,   # use bf16 to levereage GPU tensor cores
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    )

def forward_once(train_batch, model):
    device = 'cuda'
    input_ids = train_batch["input_ids"].to(device)
    labels = train_batch["labels"].to(device)

    logits = model(input_ids).logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))



def save_pretrained(output_dir, model, tokenizer):
    # Save the model weights
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


def gradient_accumulation(model, data_loader, loss_fn, optimizer):
    gradient_accumulation_steps = 4  # 假设我们想要 4 倍于显存限制的 Batch Size [cite: 227]
    for idx, (inputs, labels) in enumerate(data_loader): 
        # 1. 前向传播 (Forward Pass)
        logits = model(inputs)
        
        # 2. 关键的“缩放”操作 (Scaling)
        # 我们把当前这一小批的 Loss 除以 4 
        loss = loss_fn(logits, labels) / gradient_accumulation_steps 
        
        # 3. 梯度累加 (Backward Pass)
        # 这一步计算梯度，并把它们“加”到每个参数的 .grad 属性中 [cite: 232, 233]
        # 注意：此时并没有更新模型，只是在往容器里“倒水”
        loss.backward() 
        
        # 4. 检查是否“攒够了” (The Checkpoint)
        if (idx + 1) % gradient_accumulation_steps == 0: 
            
            # 5. 统一更新 (The Action)
            # 只有攒够了 4 次，才真正走一步优化 
            optimizer.step()
            
            # 6. 清空容器 (The Reset)
            # 更新完后，把 .grad 归零，准备迎接下一个循环 [cite: 238]
            optimizer.zero_grad()
            


def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer):
    """
    Tokenize the question and output seperately, concat them together, and 
    constructs a `response_mask`. 
    
    Return:
    dict[str, torch.Tensor]. What should the key be? 
    """
    # validity check
    assert len(prompt_strs) == len(output_strs), "input lengths do not match"
    
    
    input_ids_cache = []
    labels_cache = []
    max_len = 0
    batch_size = len(prompt_strs)

    for idx, (prompt, response) in enumerate(zip(prompt_strs, output_strs)):
        print(idx, prompt, response)
        
        # tokenize prompt and response 
        prompt_tokenized   = tokenizer(prompt, return_tensors="pt")
        response_tokenized = tokenizer(response, return_tensors="pt")
        
        print(f"prompt_tokenized result:\n{prompt_tokenized}")
        print(f"response_tokenized result:\n{response_tokenized}")
        
        prompt_ids    = prompt_tokenized['input_ids'].squeeze(0)
        response_ids  = response_tokenized['input_ids'].squeeze(0)
        prompt_mask   = prompt_tokenized['attention_mask'].squeeze(0)
        response_mask = response_tokenized['attention_mask'].squeeze(0)
        
        # get length info 
        prompt_len   = prompt_ids.shape[0]
        response_len = response_ids.shape[0]
        total_len = prompt_len + response_len
        
        # concat prompt and response 
        ids_concat = torch.concat([prompt_ids, response_ids])
        print(f"After concat, the ids look like:\n{ids_concat}")
        
        # Store the concatenated ids and length info for later padding and shifting
        input_ids_cache.append(ids_concat)
        labels_cache.append((prompt_len, response_len))
        
        # record the longest total length for later padding 
        max_len = max(max_len, total_len)
        
    # construct padding tensors aforehand
    # Note: input_ids and labels have length max_len - 1 due to shifting
    pad_id = tokenizer.pad_token_id
    input_ids_batch = torch.full((batch_size, max_len - 1), pad_id)
    labels_batch    = torch.full((batch_size, max_len - 1), pad_id)
    mask_batch      = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)
    
    # pad and shift these batch tensors
    for i in range(batch_size):
        # First pad the concatenated ids to max_len
        ids_concat = input_ids_cache[i]
        ids_padded = torch.full((max_len,), pad_id)
        ids_padded[:len(ids_concat)] = ids_concat
        
        # Then shift: input_ids = padded[:-1], labels = padded[1:]
        input_ids_batch[i, :] = ids_padded[:-1]
        labels_batch[i, :] = ids_padded[1:]
        
        # construct mask from length info
        prompt_len, response_len = labels_cache[i]
        mask_prompt = torch.zeros(prompt_len - 1)
        mask_response = torch.ones(response_len)
        curr_mask = torch.concat([mask_prompt, mask_response]).bool()
        mask_batch[i, :len(curr_mask)] = curr_mask
        
        
    print(f"\nFinal result")
    print(f"input_ids_batch:\n{input_ids_batch}")
    print(f"labels_batch:\n{input_ids_batch}")
    print(f"mask_batch:\n{mask_batch}")
    
    # construct and return 
    result = {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "response_mask": mask_batch,
    }
    return result
            
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    prompt_strs = [
        "Hello, this is ", 
        "What is the main capitcal city that has the largest population"
    ]
    output_strs = [
        "Jack speaking", 
        "of France?"
    ]
    
    print("Check the token for id==0")
    print(tokenizer.convert_ids_to_tokens([0]))
    
    
    tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)