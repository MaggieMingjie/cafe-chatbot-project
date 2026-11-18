# C:\Users\Maggi\Desktop\cafe_chatbot_project\src\model\cafe_model.py

import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Model, MT5Config
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Dict, Any
import torch.nn.functional as F


class IntentClassifier(nn.Module):
    """意图分类器"""
    
    def __init__(self, hidden_size: int, num_intents: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_intents)
    
    def forward(self, encoder_output):
        # 使用[CLS]位置的输出（第一个token）
        pooled_output = encoder_output[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_intents]
        return logits


class SlotFillingHead(nn.Module):
    """槽位填充头（BIO标注）"""
    
    def __init__(self, hidden_size: int, num_slot_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_slot_labels)
    
    def forward(self, encoder_output):
        # encoder_output: [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(encoder_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_slot_labels]
        return logits


class CafeDialogueModel(nn.Module):
    """
    咖啡店对话模型（混合架构）
    
    结合：
    1. Transformer基础模型（mT5）
    2. 意图分类头
    3. 槽位填充头
    4. 对话状态跟踪器（简化版）
    """
    
    def __init__(
        self,
        model_name: str = "google/mt5-small",
        num_intents: int = 7,
        num_slot_labels: int = 20,  # 假设有20种槽位标签（BIO格式）
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        confidence_threshold: float = 0.7
    ):
        super().__init__()
        
        self.confidence_threshold = confidence_threshold
        self.num_intents = num_intents
        
        # 1. 加载基础Transformer模型
        print(f"🔄 Loading base model: {model_name}")
        from transformers import MT5ForConditionalGeneration
        self.base_model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.encoder = self.base_model.encoder  # 只使用编码器部分
        self.hidden_size = self.base_model.config.d_model
        
        # 2. 任务特定头部
        self.intent_classifier = IntentClassifier(self.hidden_size, num_intents)
        self.slot_filling = SlotFillingHead(self.hidden_size, num_slot_labels)
        
        # 3. 对话状态跟踪器（简单的记忆机制）
        self.state_tracker = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # 4. LoRA配置（论文中提到的参数高效微调）
        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)
        
        # 5. 菜单知识库（用于规则回退）
        self.menu_knowledge = self._init_menu_knowledge()
        
        print(f"✅ Model initialization complete")
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Number of intents: {num_intents}")
        print(f"   Number of slot labels: {num_slot_labels}")
        print(f"   LoRA: {'Enabled' if use_lora else 'Disabled'}")
    
    def _apply_lora(self, rank: int, alpha: int):
        """应用LoRA参数高效微调"""
        # 这里简化处理，实际需要使用peft库
        print(f"   LoRA config: rank={rank}, alpha={alpha}")
        # 在实际实现中，这里会冻结大部分参数，只训练低秩适配器
    
    def _init_menu_knowledge(self) -> Dict:
        """初始化菜单知识库（用于规则回退）"""
        return {
            'drinks': ['latte', 'americano', 'cappuccino', 'mocha', 'oat latte'],
            'sizes': ['small', 'medium', 'large'],
            'temperatures': ['hot', 'iced'],
            'milk_types': ['whole', 'oat', 'almond', 'non-fat'],
            'add_ons': ['extra shot', 'vanilla syrup', 'whipped cream'],
            'sweetness': ['no sugar', 'half sugar', 'regular sugar', 'less sugar']
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            intent_labels: 意图标签 [batch_size]
            slot_labels: 槽位标签 [batch_size, seq_len]
            return_dict: 是否返回字典
        
        Returns:
            Dictionary containing loss and logits
        """
        # 1. 编码器前向传播
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取编码器输出 [batch_size, seq_len, hidden_size]
        sequence_output = encoder_outputs.last_hidden_state
        
        # 2. 意图分类
        intent_logits = self.intent_classifier(sequence_output)
        
        # 3. 槽位填充
        slot_logits = self.slot_filling(sequence_output)
        
        # 4. 计算损失
        total_loss = None
        if intent_labels is not None:
            intent_loss = F.cross_entropy(intent_logits, intent_labels)
            total_loss = intent_loss
        
        if slot_labels is not None and total_loss is not None:
            # 槽位损失（需要处理填充位置）
            active_loss = attention_mask.view(-1) == 1
            active_slot_logits = slot_logits.view(-1, slot_logits.size(-1))
            active_slot_labels = slot_labels.view(-1)
            slot_loss = F.cross_entropy(active_slot_logits[active_loss], active_slot_labels[active_loss])
            
            # 论文中设置的λ=1.2
            total_loss = total_loss + 1.2 * slot_loss
        
        # 5. 计算置信度（用于混合系统的决策）
        intent_probs = F.softmax(intent_logits, dim=-1)
        intent_confidence = torch.max(intent_probs, dim=-1)[0]
        
        # 6. 应用规则回退（如果置信度低于阈值）
        if not self.training and intent_confidence.mean().item() < self.confidence_threshold:
            # 这里可以添加规则回退逻辑
            pass
        
        if return_dict:
            return {
                'loss': total_loss,
                'intent_logits': intent_logits,
                'slot_logits': slot_logits,
                'intent_confidence': intent_confidence,
                'encoder_outputs': sequence_output
            }
        else:
            return total_loss, intent_logits, slot_logits
    
    def predict_with_fallback(self, input_ids, attention_mask):
        """
        带规则回退的预测（论文3.3.3节描述的混合系统）
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        
        intent_probs = F.softmax(outputs['intent_logits'], dim=-1)
        intent_confidence = outputs['intent_confidence']
        predicted_intent = torch.argmax(intent_probs, dim=-1)
        
        # 如果置信度低于阈值，使用规则回退
        if intent_confidence.item() < self.confidence_threshold:
            # 这里实现规则回退逻辑
            # 例如：检查输入中是否包含菜单项
            predicted_intent = self._rule_based_fallback(input_ids)
        
        return predicted_intent, outputs['slot_logits']
    
    def _rule_based_fallback(self, input_ids):
        """简单的规则回退"""
        # 这里可以根据实际需求实现
        # 返回默认意图
        return torch.tensor([0])  # 假设0是order意图


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = CafeDialogueModel(
        model_name="google/mt5-small",
        num_intents=7,
        num_slot_labels=20
    )
    
    # 创建模拟输入
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 前向传播
    with torch.no_grad():
        # 直接调用encoder
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = encoder_outputs.last_hidden_state
        
        # 意图分类
        intent_logits = model.intent_classifier(sequence_output)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        
        # 槽位填充
        slot_logits = model.slot_filling(sequence_output)
    
    print(f"\n📊 Model Output:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Encoder output shape: {sequence_output.shape}")
    print(f"Intent logits shape: {intent_logits.shape}")
    print(f"Intent probabilities: {intent_probs[0][:5]}...")  # 显示前5个意图的概率
    print(f"Slot logits shape: {slot_logits.shape}")