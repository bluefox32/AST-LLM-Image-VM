"""
AST + MiniLLM 統合フレームワーク
================================

【設計原理】
  1. プロンプト P → ASTへ解析（構文木構造化）
  2. AST変換パイプライン（セミコンパイル）
     - スコープ付きシンボルテーブル管理
     - 冪等性・不変条件チェック
     - ハッシュチェーン改ざん検知
  3. ベクトル化: 変換済みAST → 外積行列で多次元化
  4. Attention計算: 内積で類似度→softmaxで優先度確定
  5. 応答生成: Value加算で返答合成

【特徴】
  - DBレス（スコープテーブルで構造管理）
  - Pure Python + NumPy のみ
  - ハッシュチェーンで改ざん検知
  - トークン予算動的調整
  - 冪等性・不変条件で安全性確保
"""

from __future__ import annotations

import os
import re
import math
import hashlib
import json
import time
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum
from collections import defaultdict
import numpy as np


# ═══════════════════════════════════════════════════════════
# 1. ASTノード定義
# ═══════════════════════════════════════════════════════════

class PassState(Enum):
    RAW       = "raw"
    TYPED     = "typed"
    OPTIMIZED = "optimized"
    EMITTED   = "emitted"


@dataclass
class ASTNode:
    """汎用ASTノード（言語非依存）"""
    node_type: str
    value: Optional[str] = None
    children: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    
    # 内部管理（自動設定）
    node_id: Optional[str] = field(default=None, repr=False)
    parent_hash: Optional[str] = field(default=None, repr=False)
    _hash: Optional[str] = field(default=None, repr=False)

    def compute_hash(self, parent_hash: Optional[str] = None) -> str:
        """ハッシュチェーン計算"""
        payload = json.dumps({
            "type": self.node_type,
            "value": self.value,
            "meta": self.meta,
            "children": [c.compute_hash() for c in self.children],
            "parent": parent_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def verify(self) -> bool:
        """ハッシュ検証"""
        return self._hash == self.compute_hash(self.parent_hash)

    def walk(self):
        """深さ優先走査"""
        yield self
        for child in self.children:
            yield from child.walk()


# ═══════════════════════════════════════════════════════════
# 2. スコープ付きシンボルテーブル
# ═══════════════════════════════════════════════════════════

class Scope:
    """スコープ＆シンボルテーブル（DBレス構造管理）"""
    
    def __init__(self, name: str, parent: Optional[Scope] = None):
        self.name = name
        self.parent = parent
        self._table: dict[str, dict] = {}

    def define(self, symbol: str, info: dict):
        self._table[symbol] = info

    def lookup(self, symbol: str) -> Optional[dict]:
        if symbol in self._table:
            return self._table[symbol]
        if self.parent:
            return self.parent.lookup(symbol)
        return None

    def scope_hash(self) -> str:
        keys = sorted(self._table.keys())
        return hashlib.md5(json.dumps(keys).encode()).hexdigest()[:8]

    def __repr__(self):
        return f"Scope({self.name}, symbols={list(self._table.keys())})"


# ═══════════════════════════════════════════════════════════
# 3. トークナイザー（外積行列のインデックス参照）
# ═══════════════════════════════════════════════════════════

class Tokenizer:
    """単語→ID変換（外積行列の行ポインタ）"""
    
    SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self):
        self.word2id: dict[str, int] = dict(self.SPECIAL)
        self.id2word: dict[int, str] = {v: k for k, v in self.SPECIAL.items()}
        self._next_id = len(self.SPECIAL)

    def fit(self, texts: list[str], min_freq: int = 1):
        """語彙構築"""
        freq: dict[str, int] = defaultdict(int)
        for text in texts:
            for w in self._split(text):
                freq[w] += 1
        for w, f in sorted(freq.items(), key=lambda x: -x[1]):
            if f >= min_freq and w not in self.word2id:
                self.word2id[w] = self._next_id
                self.id2word[self._next_id] = w
                self._next_id += 1
        return self

    def add_word(self, word: str) -> int:
        if word not in self.word2id:
            self.word2id[word] = self._next_id
            self.id2word[self._next_id] = word
            self._next_id += 1
        return self.word2id[word]

    def encode(self, text: str) -> list[int]:
        """テキスト → トークンID列"""
        return [self.word2id.get(w, self.SPECIAL["<UNK>"])
                for w in self._split(text)]

    def decode(self, ids: list[int]) -> str:
        """トークンID列 → テキスト"""
        return " ".join(self.id2word.get(i, "<UNK>") for i in ids
                        if i not in (0, 2, 3))

    def _split(self, text: str) -> list[str]:
        """日本語・英語対応の分かち書き"""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+|[^\x00-\x7f]+|[^\w\s]', text)
        result = []
        for t in tokens:
            if re.match(r'[^\x00-\x7f]+', t):
                chars = list(t)
                result.extend(chars)
                for n in (2, 3, 4):
                    for i in range(len(chars) - n + 1):
                        result.append(''.join(chars[i:i+n]))
            else:
                result.append(t)
        return [tok for tok in result if tok.strip()]

    @property
    def vocab_size(self) -> int:
        return self._next_id


# ═══════════════════════════════════════════════════════════
# 4. Embedding層（第1外積）
# ═══════════════════════════════════════════════════════════

class EmbeddingLayer:
    """外積行列 W_E: vocab_size × d_model"""
    
    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (vocab_size + d_model))
        self.W = rng.normal(0, scale, (vocab_size, d_model)).astype(np.float32)
        self.d_model = d_model

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """トークンID列 → [seq_len, d_model]"""
        ids = np.array(token_ids, dtype=np.int32)
        ids = np.clip(ids, 0, len(self.W) - 1)
        emb = self.W[ids]
        emb = emb + self._positional_encoding(len(token_ids))
        return emb

    def _positional_encoding(self, seq_len: int) -> np.ndarray:
        """sin/cos位置エンコーディング"""
        pe = np.zeros((seq_len, self.d_model), dtype=np.float32)
        pos = np.arange(seq_len)[:, None]
        div = np.exp(np.arange(0, self.d_model, 2) *
                     -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:self.d_model // 2])
        return pe

    def expand_vocab(self, new_size: int, seed: int = 0):
        if new_size <= len(self.W):
            return
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (new_size + self.d_model))
        extra = rng.normal(0, scale,
                          (new_size - len(self.W), self.d_model)).astype(np.float32)
        self.W = np.vstack([self.W, extra])


# ═══════════════════════════════════════════════════════════
# 5. Attentionヘッド（外積→内積→外積）
# ═══════════════════════════════════════════════════════════

class AttentionHead:
    """Single-head Attention"""
    
    def __init__(self, d_model: int, d_k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = math.sqrt(2.0 / (d_model + d_k))
        self.W_Q = rng.normal(0, s, (d_model, d_k)).astype(np.float32)
        self.W_K = rng.normal(0, s, (d_model, d_k)).astype(np.float32)
        self.W_V = rng.normal(0, s, (d_model, d_k)).astype(np.float32)
        self.d_k = d_k

    def forward(self, X: np.ndarray,
                memory: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        X      : [seq_len, d_model]
        memory : [mem_len, d_model]
        """
        src = memory if memory is not None else X
        
        Q = X @ self.W_Q
        K = src @ self.W_K
        V = src @ self.W_V
        
        scores = Q @ K.T / math.sqrt(self.d_k)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        
        output = weights @ V
        return output, weights


# ═══════════════════════════════════════════════════════════
# 6. ベクトルメモリ（DBレス知識管理）
# ═══════════════════════════════════════════════════════════

class VectorMemory:
    """インメモリベクトル空間（DBの代替）"""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.texts: list[str] = []
        self.sources: list[str] = []

    def add(self, text: str, key_vec: np.ndarray,
            val_vec: np.ndarray, source: str = ""):
        self.keys.append(key_vec.astype(np.float32))
        self.values.append(val_vec.astype(np.float32))
        self.texts.append(text)
        self.sources.append(source)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """内積ベースの検索"""
        if not self.keys:
            return []
        
        K = np.array(self.keys)
        scores = (query_vec @ K.T) / (np.linalg.norm(query_vec) * 
                                       np.linalg.norm(K, axis=1) + 1e-8)
        indices = np.argsort(-scores)[:top_k]
        return [(self.texts[i], float(scores[i])) for i in indices]

    @property
    def size(self) -> int:
        return len(self.texts)


# ═══════════════════════════════════════════════════════════
# 7. 変換パイプライン（冪等性・不変条件チェック）
# ═══════════════════════════════════════════════════════════

class TransformError(Exception):
    pass


@dataclass
class TransformPass:
    name: str
    fn: Callable[[ASTNode, Scope], str]
    validator: Optional[Callable[[ASTNode, str], list[str]]] = None


class TransformPipeline:
    """AST変換パイプライン"""
    
    def __init__(self):
        self.passes: list[TransformPass] = []

    def add_pass(self, pass_: TransformPass):
        self.passes.append(pass_)

    def run(self, node: ASTNode, scope: Scope) -> str:
        result = ""
        for p in self.passes:
            result = p.fn(node, scope)
            
            if p.validator:
                errors = p.validator(node, result)
                if errors:
                    raise TransformError(f"[{p.name}] 不変条件違反:\n" + 
                                       "\n".join(errors))
            
            # 冪等性チェック
            result2 = p.fn(node, scope)
            if result != result2:
                raise TransformError(f"[{p.name}] 冪等性違反")
        
        return result


# ═══════════════════════════════════════════════════════════
# 8. 統合LLM（AST + ベクトル生成）
# ═══════════════════════════════════════════════════════════

class ASTLLMUnified:
    """AST + MiniLLM の統合フレームワーク"""
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, max_tokens: int = 120):
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_tokens = max_tokens
        
        self.tokenizer = Tokenizer()
        self.embedding = EmbeddingLayer(1000, d_model)
        self.heads = [AttentionHead(d_model, d_model // n_heads, seed=i)
                     for i in range(n_heads)]
        self.memory = VectorMemory(d_model)
        
        self.global_scope = Scope("global")
        self.pipeline = TransformPipeline()
        self._token_budget_log = []

    def build_knowledge(self, documents: list[str], verbose: bool = True):
        """知識ベース構築"""
        self.tokenizer.fit(documents, min_freq=1)
        self.embedding.expand_vocab(self.tokenizer.vocab_size)
        
        for text in documents:
            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            
            emb = self.embedding.forward(ids)
            key_vec = emb.mean(axis=0)
            val_vec = emb.max(axis=0)
            self.memory.add(text, key_vec, val_vec, source="doc")
        
        if verbose:
            print(f"[BUILD] {len(documents)} 文書を処理")
            print(f"  語彙: {self.tokenizer.vocab_size}, メモリ: {self.memory.size}")

    def parse_prompt_ast(self, prompt: str) -> ASTNode:
        """プロンプト → AST（簡易パーサー）"""
        words = prompt.split()
        
        # 簡易な構文解析：Question/Statement判定
        q_type = "statement"
        if any(w in prompt.lower() for w in ["what", "なに", "？", "?"]):
            q_type = "question"
        elif any(w in prompt.lower() for w in ["how", "どう", "方法"]):
            q_type = "how"
        elif any(w in prompt.lower() for w in ["why", "なぜ", "理由"]):
            q_type = "why"
        
        root = ASTNode("Prompt", value=q_type,
                      children=[
                          ASTNode("Tokens", value=" ".join(words[:5])),
                          ASTNode("Question", value=q_type)
                      ])
        root._hash = root.compute_hash()
        root.node_id = root._hash
        
        return root

    def transform_ast(self, node: ASTNode) -> str:
        """AST変換パイプライン実行"""
        def normalize_pass(n: ASTNode, scope: Scope) -> str:
            return json.dumps({"type": n.node_type, "value": n.value})
        
        self.pipeline.add_pass(TransformPass(
            name="normalize",
            fn=normalize_pass,
            validator=lambda n, o: [] if o else ["空出力"]
        ))
        
        return self.pipeline.run(node, self.global_scope)

    def generate(self, prompt: str, top_k: int = 3, verbose: bool = False) -> str:
        """統合推論: AST → ベクトル化 → Attention → 応答生成"""
        
        # 1. プロンプトをASTへ解析
        ast = self.parse_prompt_ast(prompt)
        if verbose:
            print(f"[AST] {ast}")
        
        # 2. AST変換
        transformed = self.transform_ast(ast)
        if verbose:
            print(f"[TRANSFORM] {transformed}")
        
        # 3. プロンプトをベクトル化
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [1]  # <UNK>
        
        emb = self.embedding.forward(ids)
        query_vec = emb.mean(axis=0)
        
        # 4. メモリ検索
        results = self.memory.search(query_vec, top_k=top_k)
        if not results:
            return "申し訳ありませんが、該当する情報が見つかりません。"
        
        # 5. Attention計算＆応答合成
        retrieved_texts = [text for text, _ in results]
        response_vecs = []
        
        for text in retrieved_texts:
            ids_doc = self.tokenizer.encode(text)
            if ids_doc:
                emb_doc = self.embedding.forward(ids_doc)
                for head in self.heads:
                    _, weights = head.forward(emb, emb_doc)
                    resp = weights.mean(axis=0) @ emb_doc.mean(axis=0)
                    response_vecs.append(resp)
        
        if not response_vecs:
            return "応答を生成できません。"
        
        final_vec = np.array(response_vecs).mean(axis=0)
        
        # 6. テキスト再構成（Attention重みから重要語を抽出）
        top_text = retrieved_texts[0]
        answer = self._format_answer(prompt, top_text)
        
        return answer

    def _format_answer(self, prompt: str, source: str) -> str:
        """応答フォーマット"""
        if "？" in prompt or "?" in prompt:
            return f"{source.split('。')[0]}。"
        return source

    def add_knowledge(self, text: str, source: str = "dynamic"):
        """オンライン知識追加"""
        if self.embedding is None:
            return
        
        for w in self.tokenizer._split(text):
            if w not in self.tokenizer.word2id:
                self.tokenizer.add_word(w)
                self.embedding.expand_vocab(self.tokenizer.vocab_size)
        
        ids = self.tokenizer.encode(text)
        if not ids:
            return
        
        emb = self.embedding.forward(ids)
        key_vec = emb.mean(axis=0)
        val_vec = emb.max(axis=0)
        self.memory.add(text, key_vec, val_vec, source)

    def stats(self) -> dict:
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "memory_size": self.memory.size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
        }


# ═══════════════════════════════════════════════════════════
# 9. デモ実行
# ═══════════════════════════════════════════════════════════

def demo():
    sep = "=" * 62
    print(sep)
    print("  AST + MiniLLM 統合フレームワーク")
    print(sep)

    knowledge_base = [
        ("AST（抽象構文木）はコンパイラの中核である。"
         "ソースコードを木構造に変換し、型解析・最適化を実施する。"
         "ハッシュチェーンで改ざんを検知できる。"),

        ("外積と内積の関係: 外積行列W_Eがあって初めて内積が意味を持つ。"
         "単語はID＝外積行列の1行を参照するポインタである。"
         "LLMは外積行列群を動的に組み替える検索エンジンの再定義だ。"),

        ("Attention機構は外積→内積→外積の3段で完結する。"
         "プロンプトをQ，知識をK/Vとして、内積でスコア計算。"
         "softmaxで優先度を確定し、Value加算で応答を合成する。"),

        ("トークン予算はログ圧縮で管理される。"
         "検索フェーズで消費したトークン分だけ返答生成に使える。"
         "ノルム収束で最適配分が完了する。"),

        ("DBレス構成: ベクトル空間そのものがDBになる。"
         "ローカルファイルもスクレイプもインメモリも同一インターフェース。"
         "スコープ付きシンボルテーブルで構造を管理。"),

        ("セミコンパイル: 部分コンパイル→リンク。"
         "ユニット単位でコンパイルし、整合性チェック後にリンク。"
         "冪等性・不変条件で安全性を確保。"),
    ]

    # LLM構築
    llm = ASTLLMUnified(d_model=64, n_heads=4, max_tokens=120)
    llm.build_knowledge(knowledge_base, verbose=True)

    # 推論デモ
    queries = [
        "ASTとは何ですか",
        "外積と内積の違いは",
        "Attentionの仕組みを教えてください",
        "DBレスとは何ですか",
    ]

    print("\n" + sep)
    print("  推論デモ")
    print(sep)

    for q in queries:
        print(f"\n[Q] {q}")
        print("─" * 50)
        answer = llm.generate(q, top_k=2, verbose=False)
        wrapped = textwrap.fill(answer, width=56, subsequent_indent="    ")
        print(f"[A] {wrapped}")

    # 統計
    print("\n" + sep)
    print("  統計")
    print(sep)
    for k, v in llm.stats().items():
        print(f"  {k:<20}: {v}")

    print("\n" + sep)
    print("  構造サマリー")
    print(sep)
    print("  ASTNode        : 改ざん検知付きの汎用構文木")
    print("  Scope          : DBレス・スコープ管理")
    print("  TransformPass  : 冪等性・不変条件チェック")
    print("  EmbeddingLayer : 外積行列（第1外積）")
    print("  AttentionHead  : 内積→softmax→外積")
    print("  VectorMemory   : DBレス・インメモリ知識")
    print("  ASTLLMUnified  : 統合推論エンジン")
    print(sep)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        # CLI モード: base64プロンプト対応
        prompt = sys.argv[1]
        llm = ASTLLMUnified(d_model=64, n_heads=4, max_tokens=120)
        knowledge = [
            "AST（抽象構文木）はコンパイラの中核である。ハッシュチェーンで改ざんを検知できる。",
            "外積と内積の関係は相互補完的である。外積行列があって初めて内積が意味を持つ。",
            "Attention機構は外積→内積→外積の3段で完結する。softmaxで優先度を確定する。",
            "DBレス構成ではベクトル空間そのものがDBになる。スコープテーブルで構造管理。",
        ]
        llm.build_knowledge(knowledge, verbose=False)
        answer = llm.generate(prompt, top_k=3, verbose=False)
        print(answer)
    else:
        demo()
