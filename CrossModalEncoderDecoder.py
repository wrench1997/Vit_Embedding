import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

class CrossModalEncoderDecoder(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_model_name='resnet50', decoder_model_name='gpt2'):
        super(CrossModalEncoderDecoder, self).__init__()
        
        # 文本编码器
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_tokenizer = BertTokenizer.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # 图像编码器
        self.image_encoder = models.resnet50(pretrained=True)
        # 移除ResNet的最后一个全连接层
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        image_hidden_size = 2048  # ResNet50的输出维度
        
        # 嵌入融合层
        self.fusion_layer = nn.Linear(text_hidden_size + image_hidden_size, text_hidden_size)
        
        # 解码器
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_model_name)
        self.decoder_tokenizer = GPT2Tokenizer.from_pretrained(decoder_model_name)
        
        # 线性层将融合后的嵌入转换为解码器的输入维度
        decoder_hidden_size = self.decoder.config.n_embd
        self.embedding_projection = nn.Linear(text_hidden_size, decoder_hidden_size)
        
    def forward(self, input_text, input_image):
        # 处理文本输入
        text_inputs = self.text_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        text_outputs = self.text_encoder(**text_inputs)
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # 取[CLS]标记的嵌入
        
        # 处理图像输入
        image_features = self.image_encoder(input_image)  # 输出形状: [batch_size, 2048, 1, 1]
        image_embedding = image_features.view(image_features.size(0), -1)  # 展平为 [batch_size, 2048]
        
        # 融合嵌入
        combined_embedding = torch.cat((text_embedding, image_embedding), dim=1)  # [batch_size, text+image]
        fused_embedding = self.fusion_layer(combined_embedding)  # [batch_size, text_hidden_size]
        
        # 投影到解码器的嵌入维度
        decoder_input_embedding = self.embedding_projection(fused_embedding)  # [batch_size, decoder_hidden_size]
        
        # 准备解码器输入
        # GPT-2 需要输入序列，这里假设我们使用一个特殊的起始标记
        batch_size = input_image.size(0)
        decoder_input_ids = torch.tensor([[self.decoder_tokenizer.bos_token_id]] * batch_size)
        
        # 将投影后的嵌入作为解码器的初始输入
        # GPT-2 不支持直接注入嵌入，因此需要一种方式将融合嵌入与生成的序列结合
        # 这里我们简化处理，假设解码器仅基于融合嵌入生成文本
        
        # 通过解码器生成输出
        outputs = self.decoder(input_ids=decoder_input_ids, 
                               past=None, 
                               encoder_hidden_states=decoder_input_embedding.unsqueeze(1), 
                               encoder_attention_mask=None,
                               labels=None)
        
        # 获取生成的文本
        # 在实际应用中，您需要实现一个生成函数（如贪婪搜索或束搜索）来生成完整的文本
        return outputs.logits

# 示例用法
if __name__ == "__main__":
    # 初始化模型
    model = CrossModalEncoderDecoder()
    
    # 示例文本和图像
    sample_text = ["A cat sitting on a windowsill.", "A dog playing with a ball."]
    
    # 使用随机张量作为图像输入（通常应使用预处理后的图像张量）
    sample_images = torch.randn(2, 3, 224, 224)  # [batch_size, channels, height, width]
    
    # 前向传播
    logits = model(sample_text, sample_images)
    
    print(logits.shape)  # 输出形状示例
