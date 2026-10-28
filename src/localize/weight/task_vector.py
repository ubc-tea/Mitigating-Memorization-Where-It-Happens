import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

def do_task_vector(
    model,
    unlearn_set,
    reference_set,  # Task Vector 方法本身不需要 reference_set 参与计算，但为了接口一致保留
    num_epochs=10,   # 这里 Epoch 不需要太多，只要能提取出“方向”即可
    lr=1e-2,
    batch_size=32,
    weight_decay=0.0,
    alpha=1.0,      # 控制“遗忘”的力度 (Scaling Factor)
):
    """
    Task Vector Unlearning (Model Arithmetic).
    
    Logic:
        1. Copy the model -> ft_model
        2. Finetune ft_model on unlearn_set (to learn the "direction" of this data)
        3. vector = ft_model.weights - original_model.weights
        4. new_model = original_model - alpha * vector
        
    Args:
        alpha: 缩放系数。
               alpha > 0 表示遗忘 (Negation); 
               alpha < 0 表示增强 (Addition). 
               通常设置为 0.5 到 2.0 之间。
    """
    
    device = next(model.parameters()).device
    
    # 1. 创建副本用于寻找“遗忘方向”
    # 注意：这会占用双倍显存。如果显存不足，需要先保存 state_dict 到 CPU
    print("[Task Vector] Creating model copy for fine-tuning...")
    ft_model = copy.deepcopy(model)
    ft_model.train()
    
    # 2. 在 Unlearn Set 上进行常规 Fine-tune
    # 目的：让 ft_model "过拟合" 或者是 "专门学习" 这些需要遗忘的数据
    # 从而得到指向这些数据的参数方向
    optimizer = torch.optim.AdamW(
        ft_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    loader = DataLoader(unlearn_set, batch_size=batch_size, shuffle=True)
    
    print(f"[Task Vector] Fine-tuning copy model on unlearn set for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        for batch in tqdm(loader):
            batch = batch.to(device)
            
            # 标准的 Causal LM 训练
            outputs = ft_model(batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 3. 计算 Task Vector 并应用到原模型 (Arithmetic)
    # Formula: theta_new = theta_orig - alpha * (theta_ft - theta_orig)
    print(f"[Task Vector] Calculating vector and applying negation (alpha={alpha})...")
    
    with torch.no_grad():
        for (name, param_orig), (_, param_ft) in zip(model.named_parameters(), ft_model.named_parameters()):
            # 计算差异向量 (Task Vector)
            task_vector = param_ft.data - param_orig.data
            
            # 执行减法 (Unlearning)
            # 如果 alpha 是正数，这就是在“远离” Fine-tuned 的参数分布
            param_orig.data = param_orig.data - 1000*alpha * task_vector

    # 清理显存
    del ft_model
    torch.cuda.empty_cache()
    
    print("[Task Vector] Done.")
    return model