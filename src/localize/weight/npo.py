import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from peft import LoraConfig, get_peft_model, TaskType

def get_batch_log_probs(model, batch):
    device = next(model.parameters()).device
    if isinstance(batch, dict):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None: attention_mask = attention_mask.to(device)
        labels = input_ids
    else:
        input_ids = batch.to(device)
        labels = input_ids
        attention_mask = None
    
    outputs = model(input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].contiguous()
    else:
        shift_mask = torch.ones_like(shift_labels)
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size()) * shift_mask
    return -token_losses.sum(dim=1)

def do_npo(
    model,
    unlearn_set,
    reference_set,
    num_epochs=5,
    lr=1e-5,          
    batch_size=16,
    weight_decay=0.01,
    beta=0.1,
    retain_weight=50.0,
    use_lora=True,
):
    device = next(model.parameters()).device

    # ================= Phase 1: 预计算 Reference Logits =================
    print("[NPO] Phase 1: Pre-computing reference log probabilities...")
    unlearn_loader_ref = DataLoader(unlearn_set, batch_size=batch_size, shuffle=False)
    ref_log_probs_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(unlearn_loader_ref, desc="Pre-computing"):
            log_probs = get_batch_log_probs(model, batch)
            ref_log_probs_list.append(log_probs.cpu()) 
            
    torch.cuda.empty_cache()
    gc.collect()

    # ================= Phase 2: 应用 LoRA 保护层 =================
    if use_lora:
        print("[NPO] Applying LoRA adapters to prevent model collapse...")
        try:
            # 自动推断 target_modules (如果是 GPT2, 通常是 c_attn)
            # 为了通用性，这里不硬编码 target_modules，让 PEFT 自己尝试或者使用常见的
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8,            
                lora_alpha=32, 
                lora_dropout=0.05,
                # 如果是 GPT2，modules 是 c_attn; LLaMA 是 q_proj, v_proj
                # 不写 target_modules 让 peft 自动尝试匹配，或者手动指定
                # target_modules=["c_attn"] if "gpt2" in model.config.model_type else ["q_proj", "v_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"[Warning] LoRA init failed ({e}), falling back to full fine-tuning.")

    # ================= Phase 3: 训练 =================
    print(f"[NPO] Start Training. Retain Weight: {retain_weight}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    unlearn_loader = DataLoader(unlearn_set, batch_size=batch_size, shuffle=False)
    ref_loader = DataLoader(reference_set, batch_size=batch_size, shuffle=True)
    ref_iter = iter(ref_loader)
    
    initial_retain_loss = None
    
    model.train()
    
    for epoch in range(num_epochs):
        ref_probs_iter = iter(ref_log_probs_list)
        total_npo_loss = 0
        total_retain_loss = 0
        
        for step, batch in enumerate(tqdm(unlearn_loader, desc=f"Epoch {epoch+1}")):
            try:
                retain_batch = next(ref_iter)
            except StopIteration:
                ref_iter = iter(ref_loader)
                retain_batch = next(ref_iter)
                
            # --- Retain Loss (Safety Check) ---
            if isinstance(retain_batch, dict):
                retain_batch = {k: v.to(device) for k, v in retain_batch.items()}
                out_retain = model(**retain_batch, labels=retain_batch['input_ids'])
            else:
                retain_batch = retain_batch.to(device)
                out_retain = model(retain_batch, labels=retain_batch)
            
            retain_loss = out_retain.loss
            
            # 熔断检查
            if initial_retain_loss is None:
                initial_retain_loss = retain_loss.item()
            elif retain_loss.item() > initial_retain_loss * 3.0: 
                print(f"\n[EMERGENCY STOP] Retain loss spiked to {retain_loss.item():.2f}. Stopping.")
                
                # 安全退出逻辑
                if use_lora and hasattr(model, "merge_and_unload"):
                    model = model.merge_and_unload()
                return model

            # --- NPO Loss ---
            current_ref_log_probs = next(ref_probs_iter).to(device)
            current_unlearn_log_probs = get_batch_log_probs(model, batch)
            
            u = beta * (current_ref_log_probs - current_unlearn_log_probs)
            npo_loss = -F.logsigmoid(u).mean()
            
            # --- Total Loss ---
            loss = npo_loss + retain_weight * retain_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            total_npo_loss += npo_loss.item()
            total_retain_loss += retain_loss.item()

        avg_npo = total_npo_loss / len(unlearn_loader)
        avg_retain = total_retain_loss / len(unlearn_loader)
        print(f"Epoch {epoch+1} | NPO: {avg_npo:.4f} | Retain: {avg_retain:.4f}")

    # ================= Phase 4: 合并 =================
    if use_lora:
        if hasattr(model, "merge_and_unload"):
            print("[NPO] Merging LoRA weights back to base model...")
            model = model.merge_and_unload()
        else:
            print("[NPO] Model is not a PeftModel, skipping merge.")
        
    return model