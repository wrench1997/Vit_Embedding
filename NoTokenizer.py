
import numpy as np
from collections import Counter
from typing import Optional



# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str, add_bos: bool, add_eos: bool):
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: list[int] | None = None
    ) -> tuple[list[str], list[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class PureEntropyTokenizer(Tokenizer):
    def __init__(
        self, 
        window_size: int = 8,
        min_entropy: float = 0.5,
        max_entropy: float = 4.0
    ):
        self.window_size = window_size
        self.min_entropy = min_entropy 
        self.max_entropy = max_entropy
        
        # 基础词表大小256(字节) + 特殊token
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258


        self.entropy_params = {}
        
        
    def calculate_local_entropy(self, byte_seq: bytes) -> float:
        if len(byte_seq) <= 1:
            return 0.0
        counts = Counter(byte_seq)
        probs = np.array([count/len(byte_seq) for count in counts.values()])
        return -np.sum(probs * np.log2(probs))


    def train(self, texts: list[str]):
        # 收集熵统计信息
        entropy_stats = []
        for text in texts:
            byte_seq = text.encode('utf-8')
            for i in range(len(byte_seq) - self.window_size):
                window = byte_seq[i:i+self.window_size]
                entropy = self.calculate_local_entropy(window)
                entropy_stats.append(entropy)
                
        # 学习最优参数
        self.entropy_params['optimal_window'] = self.optimize_window_size(entropy_stats)
        self.entropy_params['entropy_thresholds'] = self.learn_thresholds(entropy_stats)

    def find_entropy_boundaries(self, byte_seq: bytes) -> list[int]:
        """基于局部熵变化找分割点"""
        boundaries = [0]
        pos = 0
        
        while pos < len(byte_seq):
            # 在窗口范围内寻找最佳分割点
            max_delta = -float('inf')
            best_split = pos + 1
            
            for split in range(pos + 1, min(pos + self.window_size, len(byte_seq))):
                left = byte_seq[pos:split]
                right = byte_seq[split:split + self.window_size]
                
                if len(left) == 0 or len(right) == 0:
                    continue
                    
                left_entropy = self.calculate_local_entropy(left)
                right_entropy = self.calculate_local_entropy(right)
                
                # 计算熵变化
                entropy_delta = abs(left_entropy - right_entropy)
                
                # 根据熵的范围和变化选择分割点
                if (self.min_entropy <= left_entropy <= self.max_entropy and
                    self.min_entropy <= right_entropy <= self.max_entropy and
                    entropy_delta > max_delta):
                    max_delta = entropy_delta
                    best_split = split
            
            boundaries.append(best_split)
            pos = best_split
            
        return boundaries

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        byte_seq = text.encode('utf-8')
        boundaries = self.find_entropy_boundaries(byte_seq)
        
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
            
        # 根据边界分割并转换为token
        for start, end in zip(boundaries, boundaries[1:]):
            tokens.extend(list(byte_seq[start:end]))
            
        if add_eos:
            tokens.append(self.eos_id)
            
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode('utf-8', errors='ignore')

    def get_token_offsets(
        self, 
        text: str, 
        tokens: Optional[list[int]] = None
    ) -> tuple[list[str], list[int]]:
        if tokens is None:
            tokens = self.encode(text)
            
        decoded_chars, offsets = [], []
        byte_pos = 0
        
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode('utf-8', errors='ignore')
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode('utf-8'))
                
        return decoded_chars, offsets


tokenizer = PureEntropyTokenizer(
    window_size=8,
    min_entropy=0.5,
    max_entropy=4.0
)

text = "Hello World!"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
print(decoded)


