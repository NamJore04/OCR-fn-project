"""
Module Transformer Decoder cho OCR Image-to-Text
Nhận vào chuỗi visual tokens và sinh ra chuỗi text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Mã hóa vị trí (positional encoding) cho chuỗi token đầu vào
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Tạo mã hóa vị trí một lần
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Đăng ký như một buffer (không phải parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor đầu vào, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder cho mô hình OCR
    """
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, 
                 d_ff=2048, dropout=0.1, max_len=100):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # [batch_size, seq_len, d_model]
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Khởi tạo tham số
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Khởi tạo tham số của mô hình
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass của Transformer Decoder
        
        Args:
            tgt: Tensor token đầu vào, shape [batch_size, tgt_len]
            memory: Visual tokens từ encoder, shape [batch_size, src_len, d_model]
            tgt_mask: Mask để ẩn các vị trí trong tương lai, shape [tgt_len, tgt_len]
            tgt_key_padding_mask: Mask để ẩn các padding token, shape [batch_size, tgt_len]
        
        Returns:
            output: Logits cho từng token, shape [batch_size, tgt_len, vocab_size]
        """
        # Token embedding
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, tgt_len, d_model]
        
        # Thêm positional encoding
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        
        # Transformer Decoder
        # [batch_size, tgt_len, d_model]
        output = self.transformer_decoder(
            tgt=tgt_emb,  # Target sequence
            memory=memory,  # Memory from encoder
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Output projection
        # [batch_size, tgt_len, vocab_size]
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz, device="cpu"):
        """
        Tạo mask để ẩn đi các token trong tương lai (autoregressive)
        
        Args:
            sz: Kích thước của mask
            device: Device để tạo mask
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def decode(self, visual_tokens, start_token_id, end_token_id, max_length=None):
        """
        Decode visual tokens thành chuỗi token đầu ra
        
        Args:
            visual_tokens: Visual tokens từ encoder, shape [batch_size, src_len, d_model]
            start_token_id: ID của token bắt đầu
            end_token_id: ID của token kết thúc
            max_length: Độ dài tối đa của chuỗi sinh ra
        
        Returns:
            output_ids: Chuỗi token đầu ra, shape [batch_size, tgt_len]
        """
        if max_length is None:
            max_length = self.max_len
        
        batch_size = visual_tokens.size(0)
        device = visual_tokens.device
        
        # Khởi tạo danh sách ids đầu ra với token bắt đầu
        # [batch_size, 1]
        output_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        # Cờ đánh dấu đã kết thúc cho từng mẫu
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Vòng lặp autoregressive
        for i in range(max_length - 1):
            # Tạo mask
            tgt_mask = self.generate_square_subsequent_mask(
                output_ids.size(1),
                device=device
            )
            
            # Dự đoán token tiếp theo
            # [batch_size, cur_len, vocab_size]
            logits = self.forward(
                tgt=output_ids,
                memory=visual_tokens,
                tgt_mask=tgt_mask
            )
            
            # Lấy token có xác suất cao nhất ở vị trí cuối cùng
            # [batch_size, vocab_size] -> [batch_size]
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Thêm token mới vào chuỗi đầu ra
            # [batch_size, i+2]
            output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=1)
            
            # Kiểm tra end token
            finished = finished | (next_token == end_token_id)
            
            # Dừng lại nếu tất cả đã kết thúc
            if torch.all(finished):
                break
        
        return output_ids
    
    def beam_search_decode(self, visual_tokens, start_token_id, end_token_id, beam_size=5, max_length=None):
        """
        Decode visual tokens thành chuỗi token đầu ra sử dụng beam search
        
        Args:
            visual_tokens: Visual tokens từ encoder, shape [1, src_len, d_model]
            start_token_id: ID của token bắt đầu
            end_token_id: ID của token kết thúc
            beam_size: Kích thước beam
            max_length: Độ dài tối đa của chuỗi sinh ra
        
        Returns:
            output_ids: Chuỗi token đầu ra có xác suất cao nhất, shape [1, tgt_len]
        """
        if max_length is None:
            max_length = self.max_len
        
        # Beam search chỉ hỗ trợ batch_size=1
        assert visual_tokens.size(0) == 1, "Beam search chỉ hỗ trợ batch_size=1"
        
        device = visual_tokens.device
        
        # Khởi tạo beam với token bắt đầu
        # mỗi beam là một tuple (tokens, score, finished_flag)
        beams = [(torch.tensor([[start_token_id]], device=device), 0.0, False)]
        finished_beams = []
        
        # Vòng lặp autoregressive
        for i in range(max_length - 1):
            next_beams = []
            
            # Duyệt qua mỗi beam
            for tokens, score, finished in beams:
                # Nếu beam đã kết thúc, chỉ cần thêm vào danh sách
                if finished:
                    next_beams.append((tokens, score, finished))
                    continue
                
                # Tạo mask
                tgt_mask = self.generate_square_subsequent_mask(tokens.size(1), device=device)
                
                # Dự đoán token tiếp theo
                # [1, cur_len, vocab_size]
                logits = self.forward(
                    tgt=tokens,
                    memory=visual_tokens,
                    tgt_mask=tgt_mask
                )
                
                # Lấy xác suất của top-k tokens ở vị trí cuối cùng
                # [1, vocab_size]
                next_token_logits = logits[:, -1, :]
                
                # Lấy log probabilities
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Lấy beam_size token có xác suất cao nhất
                # [1, beam_size], [1, beam_size]
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                
                # Duyệt qua mỗi token trong top-k
                for j in range(beam_size):
                    next_token = topk_indices[0, j]
                    next_log_prob = topk_log_probs[0, j].item()
                    
                    # Tạo token mới
                    next_tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)
                    next_score = score + next_log_prob
                    
                    # Kiểm tra end token
                    next_finished = (next_token.item() == end_token_id)
                    
                    # Thêm vào danh sách beams mới
                    if next_finished:
                        finished_beams.append((next_tokens, next_score, next_finished))
                    else:
                        next_beams.append((next_tokens, next_score, next_finished))
            
            # Chọn beam_size beams có score cao nhất
            beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Nếu tất cả các beams đã kết thúc hoặc không còn beam nào
            if len(beams) == 0 or all(beam[2] for beam in beams):
                break
        
        # Kết hợp beams đã hoàn thành và chưa hoàn thành
        all_beams = finished_beams + beams
        
        # Nếu không có beam nào kết thúc, lấy beam có score cao nhất
        if all_beams:
            best_beam = sorted(all_beams, key=lambda x: x[1], reverse=True)[0]
            return best_beam[0]  # Trả về tokens của beam tốt nhất
        else:
            # Trường hợp không có beam nào, trả về start_token_id
            return torch.tensor([[start_token_id]], device=device)

if __name__ == "__main__":
    # Test Transformer Decoder
    batch_size = 2
    vocab_size = 1000
    d_model = 512
    src_len = 196  # 14x14 visual tokens
    tgt_len = 20
    
    # Tạo dữ liệu test
    visual_tokens = torch.randn(batch_size, src_len, d_model)  # [batch_size, src_len, d_model]
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))  # [batch_size, tgt_len]
    
    # Tạo module Transformer Decoder
    decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
    
    # Test forward pass
    tgt_mask = decoder.generate_square_subsequent_mask(tgt_len)
    output = decoder(tgt=tgt, memory=visual_tokens, tgt_mask=tgt_mask)
    print(f"Target shape: {tgt.shape}")
    print(f"Visual tokens shape: {visual_tokens.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test decode
    start_token_id = 0
    end_token_id = 1
    output_ids = decoder.decode(
        visual_tokens=visual_tokens,
        start_token_id=start_token_id,
        end_token_id=end_token_id
    )
    print(f"Decoded output shape: {output_ids.shape}")
    
    # Test beam search decode (chỉ với batch_size=1)
    single_visual_tokens = visual_tokens[:1]  # Lấy batch_size=1
    beam_output = decoder.beam_search_decode(
        visual_tokens=single_visual_tokens,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        beam_size=3
    )
    print(f"Beam search output shape: {beam_output.shape}")