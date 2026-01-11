"""
Tree Attention for Speculative Decoding
=======================================

Tree attention enables efficient parallel verification of multiple
candidate token sequences. Instead of verifying each candidate
independently, we construct a tree structure and use specialized
attention masks.

Key insight: Many candidates share common prefixes, so we can
share computation for the common parts.

Reference:
- "SpecInfer: Accelerating Generative Large Language Model Serving
   with Tree-based Speculative Inference and Verification"
- NVIDIA TensorRT-LLM tree attention implementation
- ReDrafter: "Recurrent Drafter for Fast Speculative Decoding"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class TreeNode:
    """Node in the speculation tree."""
    token_id: int
    depth: int
    parent_idx: Optional[int] = None
    children_indices: List[int] = None
    probability: float = 1.0

    def __post_init__(self):
        if self.children_indices is None:
            self.children_indices = []


class SpeculationTree:
    """
    Tree structure for organizing speculative token candidates.

    The tree represents all candidate token sequences, with:
    - Root: The current context (last generated token)
    - Each path from root to leaf: A candidate sequence
    - Shared prefixes are shared in the tree structure

    Example:
        For candidates: [A, B, C], [A, B, D], [A, E, F]
        Tree structure:
              A
             / \\
            B   E
           / \\   \\
          C   D   F
    """

    def __init__(self):
        self.nodes: List[TreeNode] = []
        self.node_to_idx: Dict[Tuple[int, ...], int] = {}

    def add_candidate(self, tokens: List[int], probabilities: Optional[List[float]] = None):
        """
        Add a candidate token sequence to the tree.

        Args:
            tokens: List of token IDs forming the candidate sequence
            probabilities: Optional probabilities for each token
        """
        if probabilities is None:
            probabilities = [1.0] * len(tokens)

        current_path = ()
        parent_idx = None

        for depth, (token, prob) in enumerate(zip(tokens, probabilities)):
            new_path = current_path + (token,)

            if new_path not in self.node_to_idx:
                # Create new node
                node = TreeNode(
                    token_id=token,
                    depth=depth,
                    parent_idx=parent_idx,
                    probability=prob,
                )
                node_idx = len(self.nodes)
                self.nodes.append(node)
                self.node_to_idx[new_path] = node_idx

                # Update parent's children
                if parent_idx is not None:
                    self.nodes[parent_idx].children_indices.append(node_idx)

            parent_idx = self.node_to_idx[new_path]
            current_path = new_path

    def get_attention_mask(self) -> torch.Tensor:
        """
        Generate attention mask for tree attention.

        The mask allows each node to attend to:
        - All nodes on the path from root to itself
        - Not to any nodes in other branches

        Returns:
            mask: [num_nodes, num_nodes] binary mask
                  1 = can attend, 0 = cannot attend
        """
        num_nodes = len(self.nodes)
        mask = torch.zeros(num_nodes, num_nodes)

        for i, node in enumerate(self.nodes):
            # Each node can attend to itself
            mask[i, i] = 1

            # And to all ancestors
            current_idx = node.parent_idx
            while current_idx is not None:
                mask[i, current_idx] = 1
                current_idx = self.nodes[current_idx].parent_idx

        return mask

    def get_position_ids(self) -> torch.Tensor:
        """
        Get position IDs for each node in the tree.

        Position ID = depth in tree (distance from root)

        Returns:
            position_ids: [num_nodes] tensor of position indices
        """
        return torch.tensor([node.depth for node in self.nodes])

    def get_token_ids(self) -> torch.Tensor:
        """
        Get token IDs for all nodes.

        Returns:
            token_ids: [num_nodes] tensor of token IDs
        """
        return torch.tensor([node.token_id for node in self.nodes])

    def get_paths(self) -> List[List[int]]:
        """
        Get all paths from root to leaves.

        Returns:
            List of token sequences (paths)
        """
        paths = []

        def dfs(node_idx: int, current_path: List[int]):
            node = self.nodes[node_idx]
            current_path.append(node.token_id)

            if not node.children_indices:  # Leaf node
                paths.append(current_path.copy())
            else:
                for child_idx in node.children_indices:
                    dfs(child_idx, current_path)

            current_path.pop()

        # Find root nodes (nodes with no parent)
        root_indices = [i for i, node in enumerate(self.nodes) if node.parent_idx is None]
        for root_idx in root_indices:
            dfs(root_idx, [])

        return paths


class TreeAttention(nn.Module):
    """
    Tree Attention module for parallel verification of speculative tokens.

    This module modifies standard attention to work with tree-structured
    inputs, enabling efficient verification of multiple candidate sequences
    in a single forward pass.

    The key modifications:
    1. Position IDs are based on tree depth, not sequence position
    2. Attention mask restricts attention to ancestor nodes only
    3. All candidates are processed in parallel with proper masking
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        tree_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with tree attention.

        Args:
            hidden_states: [batch, num_nodes, hidden_size]
            tree_mask: [num_nodes, num_nodes] attention mask from tree
            position_ids: [num_nodes] position indices (tree depths)

        Returns:
            output: [batch, num_nodes, hidden_size]
        """
        batch_size, num_nodes, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply tree mask
        # Expand mask for batch and heads: [1, 1, num_nodes, num_nodes]
        tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights.masked_fill(tree_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        return output


class TreeVerifier:
    """
    Verifies speculative token sequences using tree attention.

    Given a set of candidate sequences, constructs a speculation tree
    and verifies all candidates in parallel using the target model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def build_tree(
        self,
        candidates: List[List[int]],
        probabilities: Optional[List[List[float]]] = None,
    ) -> SpeculationTree:
        """
        Build speculation tree from candidate sequences.

        Args:
            candidates: List of candidate token sequences
            probabilities: Optional probabilities for each token

        Returns:
            SpeculationTree containing all candidates
        """
        tree = SpeculationTree()

        for i, tokens in enumerate(candidates):
            probs = probabilities[i] if probabilities else None
            tree.add_candidate(tokens, probs)

        return tree

    @torch.no_grad()
    def verify(
        self,
        prefix_ids: torch.Tensor,
        candidates: List[List[int]],
        draft_probabilities: Optional[List[List[float]]] = None,
    ) -> Tuple[List[int], int, Dict[str, Any]]:
        """
        Verify candidates and return the best accepted sequence.

        Args:
            prefix_ids: Input token IDs [1, prefix_len]
            candidates: List of candidate token sequences
            draft_probabilities: Probabilities from draft model

        Returns:
            Tuple of (accepted_tokens, num_accepted, verification_info)
        """
        if not candidates:
            return [], 0, {}

        # Build speculation tree
        tree = self.build_tree(candidates, draft_probabilities)

        # Get tree tensors
        tree_tokens = tree.get_token_ids().to(self.device)
        tree_mask = tree.get_attention_mask().to(self.device)
        tree_positions = tree.get_position_ids().to(self.device)

        # Prepare input: prefix + all tree tokens
        # For simplicity, we process candidates one by one
        # (full tree attention would require model modification)

        best_path = []
        best_length = 0

        paths = tree.get_paths()
        for path in paths:
            # Verify this path
            candidate_ids = torch.tensor(path, device=self.device).unsqueeze(0)
            full_input = torch.cat([prefix_ids, candidate_ids], dim=1)

            outputs = self.model(input_ids=full_input)
            logits = outputs.logits[0]

            # Check agreement at each position
            num_accepted = 0
            prefix_len = prefix_ids.shape[1]

            for i, token in enumerate(path):
                pred_logits = logits[prefix_len + i - 1]  # Logits for this position
                pred_token = torch.argmax(pred_logits).item()

                if pred_token == token:
                    num_accepted += 1
                else:
                    break

            if num_accepted > best_length:
                best_length = num_accepted
                best_path = path[:num_accepted]

        # If all tokens in a path accepted, try to get bonus token
        if best_length == len(paths[0]):
            full_input = torch.cat([
                prefix_ids,
                torch.tensor(best_path, device=self.device).unsqueeze(0)
            ], dim=1)
            outputs = self.model(input_ids=full_input)
            bonus_token = torch.argmax(outputs.logits[0, -1]).item()
            best_path.append(bonus_token)
            best_length += 1

        return best_path, best_length, {
            "num_candidates": len(candidates),
            "tree_nodes": len(tree.nodes),
            "paths_evaluated": len(paths),
        }


def create_tree_attention_mask(
    candidates: List[List[int]],
    prefix_length: int,
) -> torch.Tensor:
    """
    Create an attention mask for tree-structured candidates.

    This is a utility function for creating attention masks that can be
    used with standard transformer implementations.

    Args:
        candidates: List of candidate token sequences
        prefix_length: Length of the shared prefix

    Returns:
        attention_mask: [total_tokens, total_tokens] mask
    """
    # Build tree to get structure
    tree = SpeculationTree()
    for tokens in candidates:
        tree.add_candidate(tokens)

    # Get tree attention mask
    tree_mask = tree.get_attention_mask()

    # Create full mask including prefix
    total_length = prefix_length + len(tree.nodes)
    full_mask = torch.zeros(total_length, total_length)

    # Prefix tokens can attend to all previous prefix tokens (causal)
    for i in range(prefix_length):
        full_mask[i, :i + 1] = 1

    # Tree tokens can attend to all prefix tokens
    full_mask[prefix_length:, :prefix_length] = 1

    # Tree tokens attend according to tree structure
    full_mask[prefix_length:, prefix_length:] = tree_mask

    return full_mask


def visualize_tree(tree: SpeculationTree) -> str:
    """
    Create ASCII visualization of speculation tree.

    Args:
        tree: SpeculationTree to visualize

    Returns:
        String representation of the tree
    """
    if not tree.nodes:
        return "Empty tree"

    lines = []

    def draw_node(node_idx: int, prefix: str, is_last: bool):
        node = tree.nodes[node_idx]
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}[{node.token_id}] (depth={node.depth})")

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_idx in enumerate(node.children_indices):
            is_last_child = (i == len(node.children_indices) - 1)
            draw_node(child_idx, child_prefix, is_last_child)

    # Find root nodes
    roots = [i for i, node in enumerate(tree.nodes) if node.parent_idx is None]

    for i, root_idx in enumerate(roots):
        is_last_root = (i == len(roots) - 1)
        draw_node(root_idx, "", is_last_root)

    return "\n".join(lines)
