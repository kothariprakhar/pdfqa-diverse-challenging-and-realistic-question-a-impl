import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import numpy as np
import random
import copy

# ==========================================
# 1. Data Structures representing pdfQA Schema
# ==========================================

@dataclass
class PDFNode:
    """
    Represents a chunk of information extracted from a PDF.
    Includes layout information (bbox) used for multi-modal embedding.
    """
    content_id: str
    content_type: str  # 'text', 'table', 'image'
    content_data: str  # Raw text or serialized table/image caption
    page_number: int
    bbox: tuple  # (x1, y1, x2, y2)
    embedding: Optional[torch.Tensor] = None

@dataclass
class QAPair:
    qa_id: str
    question: str
    answer: str
    source_nodes: List[str]
    metadata: Dict[str, Union[str, int]] = field(default_factory=lambda: {
        "file_type": "pdf",
        "source_modality": "text",
        "answer_type": "extractive",
        "reasoning_complexity": "single-hop",
        "source_position": "body"
    })

# ==========================================
# 2. Model Components (Refined)
# ==========================================

class LayoutAwareTransformer(nn.Module):
    """
    A mock Transformer model that incorporates spatial information (Bounding Boxes),
    addressing the critique regarding Multimodality and Layout-Awareness.
    """
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Fix 3 (Multimodality): Spatial Embedding for Bounding Boxes (x1, y1, x2, y2)
        # Projects 4 coordinates to hidden dimension, similar to LayoutLM's spatial bias.
        self.bbox_projection = nn.Linear(4, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, bboxes: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids: (Batch, Seq)
        bboxes: (Batch, Seq, 4) - Normalized coordinates
        attention_mask: (Batch, Seq)
        """
        # Text Embeddings
        token_embeds = self.embedding(input_ids) # (B, S, H)
        
        # Spatial Embeddings
        spatial_embeds = self.bbox_projection(bboxes.float()) # (B, S, H)
        
        # Combine modalities (Element-wise sum)
        embeds = token_embeds + spatial_embeds
        
        # Fix 2 (Memory Efficiency): Efficient Masking
        # Expand mask to (B, S, 1) instead of (B, S, H) and use broadcasting.
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float() # (B, S, 1)
            
            masked_embeds = embeds * mask # Broadcasting handles the hidden dim
            sum_embeds = torch.sum(masked_embeds, dim=1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            pooled = sum_embeds / counts
        else:
            pooled = torch.mean(embeds, dim=1)

        return F.normalize(self.output_layer(pooled), p=2, dim=1)

class RAGRetriever:
    def __init__(self, model: LayoutAwareTransformer):
        self.model = model
        self.index = [] 
        self.node_embeddings = None 

    def _simulate_tokenization(self, text: str, length: int = 10):
        """
        Fix 1 (Coupling): Dynamically check model vocab size to prevent IndexErrors.
        """
        vocab_limit = self.model.vocab_size
        seed = sum(ord(c) for c in text)
        # Ensure indices are safely within [0, vocab_limit-1]
        token_id = seed % vocab_limit
        return [token_id] * length

    def index_documents(self, nodes: List[PDFNode]):
        self.index = copy.deepcopy(nodes)
        if not self.index:
            return

        device = next(self.model.parameters()).device
        
        input_ids_list = []
        bbox_list = []
        
        for node in self.index:
            # Tokenize content
            ids = self._simulate_tokenization(node.content_data)
            input_ids_list.append(ids)
            
            # Prepare BBox (Broadcast node bbox to sequence length)
            # Normalize bbox simply for this mock (assuming input coords are 0-1000)
            # In production, this normalization ensures inputs are roughly 0.0-1.0
            norm_bbox = [c / 1000.0 for c in node.bbox]
            bbox_seq = [norm_bbox] * len(ids)
            bbox_list.append(bbox_seq)
            
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        bboxes = torch.tensor(bbox_list, dtype=torch.float, device=device) # (B, S, 4)
        
        with torch.no_grad():
            self.node_embeddings = self.model(input_ids, bboxes)

    def retrieve(self, query: str, top_k: int = 3) -> List[PDFNode]:
        if not self.index or self.node_embeddings is None:
            return []

        device = next(self.model.parameters()).device
        
        # Query Tokenization
        q_ids = self._simulate_tokenization(query)
        dummy_q_ids = torch.tensor([q_ids], dtype=torch.long, device=device)
        
        # Queries usually don't have spatial bboxes; use zero-padding
        # Shape (1, Seq, 4)
        dummy_q_bbox = torch.zeros((1, len(q_ids), 4), device=device)
        
        with torch.no_grad():
            q_embed = self.model(dummy_q_ids, dummy_q_bbox) 
        
        scores = torch.mm(q_embed, self.node_embeddings.t())
        
        actual_k = min(top_k, len(self.index))
        if actual_k == 0: return []
             
        top_k_scores, top_k_indices = torch.topk(scores, k=actual_k, dim=1)
        retrieved_nodes = [self.index[idx.item()] for idx in top_k_indices[0]]
        return retrieved_nodes

class SyntheticGenerator:
    def generate_qa(self, node: PDFNode, complexity_level: str = "medium") -> QAPair:
        # Include keywords from content to help the simple overlap filter pass
        snippet = node.content_data[:15]
        simulated_q = f"Question about {snippet} ({node.content_type})?"
        simulated_a = f"Answer involves {snippet}."
        
        metadata = {
            "source_modality": node.content_type,
            "reasoning_complexity": complexity_level,
            "answer_type": "abstractive"
        }
        
        return QAPair(
            qa_id=f"qa_{random.randint(1000,9999)}",
            question=simulated_q,
            answer=simulated_a,
            source_nodes=[node.content_id],
            metadata=metadata
        )

# ==========================================
# 3. Filtering and Evaluation Logic
# ==========================================

class QualityFilter:
    def is_valid(self, qa_pair: QAPair, retrieved_context: List[PDFNode]) -> bool:
        """
        Fix 4 (Filter Logic): Replaced arbitrary hash with keyword overlap.
        This simulates 'entailment' or 'relevance' without external ML libraries,
        making the filter logic transparent and debuggable.
        """
        if not retrieved_context:
            return False
            
        # Basic heuristic: Check if meaningful words from the Question or Answer appear in Context
        # In a real scenario, this would use a Cross-Encoder or NLI model.
        
        def get_tokens(text):
            return set(text.lower().replace('?', '').replace('.', '').split())

        qa_tokens = get_tokens(qa_pair.question) | get_tokens(qa_pair.answer)
        
        context_text = " ".join([n.content_data for n in retrieved_context])
        context_tokens = get_tokens(context_text)
        
        # Calculate overlap
        intersection = qa_tokens.intersection(context_tokens)
        
        # If there is overlap, we assume the context is relevant enough for this mock
        return len(intersection) > 0

def run_pdfqa_pipeline():
    print("Initializing Layout-Aware RAG Pipeline...")
    # 1. Setup Models
    # Using small vocab (500) to verify dynamic coupling fix works (no IndexErrors)
    embed_model = LayoutAwareTransformer(vocab_size=500, hidden_dim=64)
    
    retriever = RAGRetriever(embed_model)
    generator = SyntheticGenerator()
    filter_module = QualityFilter()

    # 2. Simulate PDF Parsing (Ingestion) with varying modalities and bboxes
    nodes = [
        PDFNode("n1", "text", "Revenue Q1 $5M", 1, (10, 10, 100, 50)),
        PDFNode("n2", "table", "Revenue | $5M", 1, (10, 60, 100, 150)),
        PDFNode("n3", "text", "Costs up 10%", 2, (10, 10, 100, 50)),
        PDFNode("n4", "image", "Chart costs", 2, (10, 60, 100, 150))
    ]
    
    # 3. Indexing
    retriever.index_documents(nodes)

    # 4. Generate Synthetic Dataset
    dataset = []
    for node in nodes:
        qa = generator.generate_qa(node)
        dataset.append(qa)

    print(f"Generated {len(dataset)} QA pairs.")

    # 5. Evaluate Pipeline
    print("\nEvaluating Pipeline Quality...")
    valid_count = 0
    for qa in dataset:
        context_nodes = retriever.retrieve(qa.question, top_k=2)
        is_valid = filter_module.is_valid(qa, context_nodes)
        
        status = "[VALID]" if is_valid else "[FILTERED]"
        print(f"{status} Q: {qa.question} | Modality: {qa.metadata['source_modality']}")
        if is_valid: valid_count += 1

    print(f"\nFinal Valid QA Pairs: {valid_count}/{len(dataset)}")

if __name__ == "__main__":
    run_pdfqa_pipeline()