import torch
import torch.nn.functional as F

import logging

from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        logger.debug(f"{idx} {prompt} {response}")
        
        # tokenize prompt and response 
        prompt_tokenized   = tokenizer(prompt, return_tensors="pt")
        response_tokenized = tokenizer(response, return_tensors="pt")
        
        logger.debug(f"prompt_tokenized result:\n{prompt_tokenized}")
        logger.debug(f"response_tokenized result:\n{response_tokenized}")
        
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
        logger.debug(f"After concat, the ids look like:\n{ids_concat}")
        
        # Store the concatenated ids and length info for later padding and shifting
        input_ids_cache.append(ids_concat)
        labels_cache.append((prompt_len, response_len))
        
        # record the longest total length for later padding 
        max_len = max(max_len, total_len)
        
    # Construct padding tensors of length max_len (before shift)
    # Strategy: build the full matrix first, then shift at the end
    pad_id = tokenizer.pad_token_id
    input_ids_batch = torch.full((batch_size, max_len), pad_id)
    labels_batch    = torch.full((batch_size, max_len), pad_id)
    mask_batch      = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # Fill the batch tensors with concatenated ids and masks
    for i in range(batch_size):
        ids_concat = input_ids_cache[i]
        # Fill in the concatenated ids directly (no shift yet)
        input_ids_batch[i, :len(ids_concat)] = ids_concat
        labels_batch[i, :len(ids_concat)] = ids_concat
        
        # Construct mask: set True for response positions
        # After shift, mask should indicate response positions in the input_ids domain
        # Response in original concat: positions [prompt_len, prompt_len+response_len)
        # After shift (input_ids = concat[:-1]): response prediction starts at position prompt_len-1
        prompt_len, response_len = labels_cache[i]
        mask_batch[i, prompt_len - 1:prompt_len + response_len - 1] = True
        
    logger.debug(f"\nFinal result (before shift):")
    logger.debug(f"input_ids_batch:\n{input_ids_batch}")
    logger.debug(f"labels_batch:\n{labels_batch}")
    logger.debug(f"mask_batch:\n{mask_batch}")
    
    # Shift at the end: input_ids = batch[:-1], labels = batch[1:], mask = batch[:-1]
    result = {
        "input_ids": input_ids_batch[:, :-1],
        "labels": labels_batch[:, 1:],
        "response_mask": mask_batch[:, :-1],
    }
    return result

def compute_entropy(logits:torch.Tensor) -> torch.Tensor:
    """
    logits has the shape: (batch_size,sequence_length,vocab_size)
    """
    log_p = torch.log_softmax(logits, dim=-1)
    entropy = -(log_p.exp() * log_p).sum(dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Shape of tensors:
    1. input_ids: (batch_size, seq_len)
    2. labels: (batch_size, seq_len)
    """
    model_outputs = model(input_ids=input_ids)
    logits = model_outputs.logits if hasattr(model_outputs, "logits") else model_outputs[0]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {
        "log_probs": token_log_probs,
    }
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits=logits)
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
    
    logger.debug("Check the token for id==0")
    logger.debug(tokenizer.convert_ids_to_tokens([0]))
    
    
    tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)