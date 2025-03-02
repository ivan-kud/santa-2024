from collections import Counter, defaultdict
import dataclasses
import datetime
import gc
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Optional, Literal, Tuple, List, Dict

import numpy as np
import torch

import generate
import tokenizer as gemma_tokenizer


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Node:
    parent_node: 'Node' = None
    tokens: Dict[int, int] = dataclasses.field(default_factory=dict)
    vocab: List[int] = dataclasses.field(default_factory=list)
    token_sequence: List[int] = dataclasses.field(default_factory=list)
    token_mean_loss_sequence: List[float] = dataclasses.field(default_factory=list)
    seq_mean_losses: List[float] = dataclasses.field(default_factory=list)


def special_token_map(
        permuted_tokens: List[int],
        tokenizer: gemma_tokenizer.Tokenizer,
        ) -> Tuple[List[int], List[str],
                   Dict[int, List[int]], Dict[str, List[str]],
                   Dict[int, List[int]], Dict[str, List[str]],
                   Dict[int, int], Dict[str, str],
                   Dict[int, int], Dict[str, str]]:
    """Initialize maps for:
    - non-single-token words;
    - no-space tokens (to use them after BOS token);
    - no-space extra token (if exists) (to use after BOS and the next token).
    """
    # Map for non-single-token words
    prev_token = tokenizer.bos_id
    prev_token_str = "<bos>"
    spaced_tokens = []
    spaced_tokens_str = []
    non_single_token_map = defaultdict(list)
    non_single_token_str_map = defaultdict(list)
    reverse_non_single_token_map = defaultdict(list)
    reverse_non_single_token_str_map = defaultdict(list)
    for token in permuted_tokens:
        token_str = tokenizer.decode([token])
        if token_str.startswith(" "):
            spaced_tokens.append(token)
            spaced_tokens_str.append(token_str)
        else:
            non_single_token_map[prev_token].append(token)
            non_single_token_str_map[prev_token_str].append(token_str)
            reverse_non_single_token_map[token].append(prev_token)
            reverse_non_single_token_str_map[token_str].append(prev_token_str)

            # Case when prev_token is used after BOS
            if prev_token_str.startswith(" "):
                no_space_prev_token_str = prev_token_str[1:]
                no_space_prev_token_list = tokenizer.encode(no_space_prev_token_str, bos=False, eos=False)
                assert len(no_space_prev_token_list) == 1
                no_space_prev_token = no_space_prev_token_list[0]
                non_single_token_map[no_space_prev_token].append(token)
                non_single_token_str_map[no_space_prev_token_str].append(token_str)
                reverse_non_single_token_map[token].append(no_space_prev_token)
                reverse_non_single_token_str_map[token_str].append(no_space_prev_token_str)

        prev_token = token
        prev_token_str = token_str

    # Map for no_space_token is used to get original spaced token after BOS.
    # Map for no_space_extra_token is used to add extra token to tokens dict when token after BOS
    # is processing (if this extra token exists).
    no_space_token_map = {}
    no_space_token_str_map = {}
    no_space_extra_token_map = {}
    no_space_extra_token_map_str = {}
    for token in permuted_tokens:
        token_str = tokenizer.decode([token])
        if token_str.startswith(" "):
            no_space_token_list = tokenizer.encode(token_str[1:], bos=False, eos=False)
            assert 1 <= len(no_space_token_list) <= 2
            no_space_token_0 = no_space_token_list[0]
            no_space_token_str_0 = tokenizer.decode([no_space_token_0])
            no_space_token_map[no_space_token_0] = token
            no_space_token_str_map[no_space_token_str_0] = token_str
            non_single_token_map[tokenizer.bos_id].append(no_space_token_0)
            non_single_token_str_map['<bos>'].append(no_space_token_str_0)
            reverse_non_single_token_map[no_space_token_0].append(tokenizer.bos_id)
            reverse_non_single_token_str_map[no_space_token_str_0].append('<bos>')
            if len(no_space_token_list) == 2:
                no_space_token_1 = no_space_token_list[1]
                no_space_token_str_1 = tokenizer.decode([no_space_token_1])
                no_space_extra_token_map[no_space_token_0] = no_space_token_1
                no_space_extra_token_map_str[no_space_token_str_0] = no_space_token_str_1
                non_single_token_map[no_space_token_0].append(no_space_token_1)
                non_single_token_str_map[no_space_token_str_0].append(no_space_token_str_1)
                reverse_non_single_token_map[no_space_token_1].append(no_space_token_0)
                reverse_non_single_token_str_map[no_space_token_str_1].append(no_space_token_str_0)

    # Use qniuqe values
    spaced_tokens = list(set(spaced_tokens))
    spaced_tokens_str = list(set(spaced_tokens_str))
    for key in non_single_token_map.keys():
        non_single_token_map[key] = list(set(non_single_token_map[key]))
    for key in non_single_token_str_map.keys():
        non_single_token_str_map[key] = list(set(non_single_token_str_map[key]))
    for key in reverse_non_single_token_map.keys():
        reverse_non_single_token_map[key] = list(set(reverse_non_single_token_map[key]))
    for key in reverse_non_single_token_str_map.keys():
        reverse_non_single_token_str_map[key] = list(set(reverse_non_single_token_str_map[key]))

    return (spaced_tokens, spaced_tokens_str,
            dict(non_single_token_map), dict(non_single_token_str_map),
            dict(reverse_non_single_token_map), dict(reverse_non_single_token_str_map),
            no_space_token_map, no_space_token_str_map,
            no_space_extra_token_map, no_space_extra_token_map_str)


class Generator:
    def __init__(
            self,
            initial_string: str,
            permuted_string: str,
            model_variant: str,
            model_path: str,
            tokenizer_path: str,
            log_file_name: str,
            device: torch.device,
            max_tree_depth: int,
            top_p: float,
            ):
        self.initial_string = initial_string
        self.permuted_string = " " + permuted_string
        self.model_variant = model_variant
        self.model_path = model_path
        self.log_file_name = log_file_name
        self.device = device
        self.max_tree_depth = max_tree_depth
        self.top_p = top_p
        
        # Load tokenizer
        self.tokenizer = gemma_tokenizer.Tokenizer(tokenizer_path)

        # Initialize permuted tokens
        self.permuted_tokens = self.tokenizer.encode(self.permuted_string, bos=False, eos=False)

        # Initialize map for non-single-token words
        (
            self.spaced_tokens, self.spaced_tokens_str,
            self.non_single_token_map, self.non_single_token_str_map,
            self.reverse_non_single_token_map, self.reverse_non_single_token_str_map,
            self.no_space_token_map, self.no_space_token_str_map,
            self.no_space_extra_token_map, self.no_space_extra_token_map_str,
            ) = special_token_map(self.permuted_tokens, self.tokenizer)

        # Initialize max_vocab_len
        self.max_vocab_len = 128

        # Initialize max sequence length and max batch size
        self.len2batch_map = {8: 4096, 16: 2048, 24: 1344, 32: 1024,
                         40: 896, 48: 768, 56: 640, 64: 512,
                         72: 480, 80: 448, 88: 416, 96: 384,
                         104: 352, 112: 320, 120: 288, 128: 256}
        self.max_seq_len = 16
        self.max_batch_size = self.len2batch_map[self.max_seq_len]

        # Load model and compile
        self.model = generate.load_model(self.model_path, self.model_variant, self.device)
        self.model = torch.compile(self.model, mode='reduce-overhead', fullgraph=True)

    def create_child_node(
            self,
            parent_node: Node,
            token: int,
            token_mean_loss: float,
            seq_mean_losses: List[float],
            seq_mean_loss: float,
            ) -> Node:
        # Create child node by copying parent node
        node = Node(
            parent_node=parent_node,
            tokens=parent_node.tokens.copy(),
            token_sequence=parent_node.token_sequence.copy(),
            token_mean_loss_sequence=parent_node.token_mean_loss_sequence.copy(),
            seq_mean_losses=seq_mean_losses,
            )

        # Update sequence
        node.token_sequence.append(token)
        node.token_mean_loss_sequence.append(token_mean_loss)

        # Update tokens counter. Cases: node after BOS; else
        if len(parent_node.token_sequence) == 1:
            # Drop original (spaced) token
            original_spaced_token = self.no_space_token_map[token]
            node.tokens[original_spaced_token] -= 1
            if node.tokens[original_spaced_token] == 0:
                del node.tokens[original_spaced_token]

            # Add extra token (if exists)
            if token in self.no_space_extra_token_map:
                no_space_extra_token = self.no_space_extra_token_map[token]
                if no_space_extra_token in node.tokens:
                    node.tokens[no_space_extra_token] += 1
                else:
                    node.tokens[no_space_extra_token] = 1
        else:
            # Drop token
            node.tokens[token] -= 1
            if node.tokens[token] == 0:
                del node.tokens[token]

        # Update sequence for non-single tokens
        while (token in self.non_single_token_map
               and len(self.non_single_token_map[token]) == 1):
            token = self.non_single_token_map[token][0]
            
            # Update sequence
            node.token_sequence.append(token)
            node.token_mean_loss_sequence.append(token_mean_loss)

            # Drop token
            node.tokens[token] -= 1
            if node.tokens[token] == 0:
                del node.tokens[token]

        # Check for EOS
        if ((len(node.tokens) == 1 and node.tokens[list(node.tokens.keys())[0]] == 1)
            or (len(node.tokens) == 0 and node.token_sequence[-1] != self.tokenizer.eos_id)):
            if len(node.tokens) == 1:
                # Add last token
                last_token = list(node.tokens.keys())[0]
                node.token_sequence.append(last_token)
                node.token_mean_loss_sequence.append(token_mean_loss)

            # Add EOS token
            node.token_sequence.append(self.tokenizer.eos_id)
            node.token_mean_loss_sequence.append(token_mean_loss)

            # Create token set and vocab as a additional EOS token
            node.tokens = {self.tokenizer.eos_id: 1}
            node.vocab = [self.tokenizer.eos_id]

        elif len(node.tokens) == 0:
            # Drop additional EOS token
            node.token_sequence = node.token_sequence[:-1]
            node.token_mean_loss_sequence = node.token_mean_loss_sequence[:-1]

            # Set vocab to empty
            node.vocab = []

        else:
            # Create vocab
            if token in self.non_single_token_map:
                node.vocab = self.non_single_token_map[token]
            else:
                node.vocab = list(set(node.tokens.keys()) & set(self.spaced_tokens))

        return node

    def get_child_node_batch(self, parent_node_batch: List[Node]) -> List[Node]:
        # Generate top tokens
        print('Batch...')
        logger.info('Batch...')
        (top_tokens_batch, top_tokens_mean_losses_batch,
         seq_mean_losses_batch, seq_mean_loss_batch) = generate.generate(
            [node.token_sequence for node in parent_node_batch],
            [node.vocab for node in parent_node_batch],
            self.model, 
            self.device,
            self.tokenizer.pad_id,
            self.max_batch_size,
            self.max_seq_len,
            self.max_vocab_len,
            self.top_p,
            )

        child_nodes = []
        for (parent_node, top_tokens, top_tokens_mean_losses,
             seq_mean_losses, seq_mean_loss) in zip(
                 parent_node_batch, top_tokens_batch, top_tokens_mean_losses_batch,
                 seq_mean_losses_batch, seq_mean_loss_batch,
                 ):
            child_node_batch = [self.create_child_node(
                parent_node, token, token_mean_loss,
                seq_mean_losses, seq_mean_loss,
                ) for token, token_mean_loss in zip(top_tokens, top_tokens_mean_losses)
                if token != self.tokenizer.pad_id]
            child_nodes.extend(child_node_batch)

            # Check for forbidden sequences
            for token in top_tokens:
                prev_token = parent_node.token_sequence[-1]
                if token in self.reverse_non_single_token_map:
                    assert prev_token in self.reverse_non_single_token_map[token]

        return child_nodes

    def get_child_nodes(self, parent_nodes: List[Node]) -> List[Node]:
        parent_node_batch: List[Node] = []
        child_nodes: List[Node] = []
        for parent_node in parent_nodes:
            parent_node_batch.append(parent_node)
            if len(parent_node_batch) == self.max_batch_size:
                child_node_batch = self.get_child_node_batch(parent_node_batch)
                child_nodes.extend(child_node_batch)
                parent_node_batch = []
        if parent_node_batch:
            child_node_batch = self.get_child_node_batch(parent_node_batch)
            child_nodes.extend(child_node_batch)
        
        return child_nodes

    def generate_sequence(self) -> Tuple[str, float]:
        # Initialize buffers for finished nodes
        buffer_size = 10_000, 128
        buffer_file_token = Path('./logs') / f'{self.log_file_name} token.npy'
        buffer_file_loss = Path('./logs') / f'{self.log_file_name} loss.npy'
        if buffer_file_token.is_file():
            # Load buffers for finished nodes
            token_sequence_buffer = np.load(buffer_file_token)
            seq_mean_losses_buffer = np.load(buffer_file_loss)
            buffer_pointer = np.sum(token_sequence_buffer == self.tokenizer.pad_id, axis=1).argmax()
        else:
            # Initialize buffers for finished nodes
            token_sequence_buffer = np.full(buffer_size, self.tokenizer.pad_id, dtype=np.int32)
            seq_mean_losses_buffer = np.full(buffer_size, float('+inf'), dtype=np.float32)        
            buffer_pointer = 0

        # Initialize current root node
        root_tokens_file = Path('./logs') / f'{self.log_file_name} root_tokens.json'
        root_vocab_file = Path('./logs') / f'{self.log_file_name} root_vocab.json'
        root_token_sequence_file = Path('./logs') / f'{self.log_file_name} root_token.npy'
        root_mean_loss_sequence_file = Path('./logs') / f'{self.log_file_name} root_loss.npy'
        if root_token_sequence_file.is_file():
            # Load root node from files
            with open(root_tokens_file) as file:
                tokens = json.load(file, object_hook=lambda x: {int(k): v for k, v in x.items()})
            with open(root_vocab_file) as file:
                vocab = json.load(file)
            current_root_node = Node(
                tokens=tokens,
                vocab=vocab,
                token_sequence=np.load(root_token_sequence_file).tolist(),
                token_mean_loss_sequence=np.load(root_mean_loss_sequence_file).tolist(),
                )
        else:
            if self.initial_string:
                initial_tokens = self.tokenizer.encode(" " + self.initial_string, bos=False, eos=False)
                tokens = dict(Counter(self.permuted_tokens) - Counter(initial_tokens))
                vocab = list(set(tokens.keys()) & set(self.spaced_tokens))
                token_sequence = self.tokenizer.encode(self.initial_string, bos=True, eos=False)
            else:
                tokens = dict(Counter(self.permuted_tokens))
                vocab = self.non_single_token_map[self.tokenizer.bos_id]
                token_sequence = [self.tokenizer.bos_id]

            # Initialize working nodes as a root node
            current_root_node = Node(
                tokens=tokens,
                vocab=vocab,
                token_sequence=token_sequence,
                token_mean_loss_sequence=[0.0] * len(token_sequence),
                )

        tree_depth = 0
        working_nodes = [current_root_node]
        prev_new_node_n = len(working_nodes)
        total_start_time = time.monotonic()
        while working_nodes:
            start_time = time.monotonic()
            new_nodes = self.get_child_nodes(working_nodes)
            tree_depth += 1

            # Separate new nodes on working and finished nodes
            finished_node_n = 0
            working_nodes: List[Node] = []
            for node in new_nodes:
                if node.vocab:
                    working_nodes.append(node)
                else:
                    # Store results of finished nodes to buffers
                    token_sequence_buffer[buffer_pointer, :len(node.token_sequence)] = node.token_sequence
                    seq_mean_losses_buffer[buffer_pointer, :len(node.seq_mean_losses)] = node.seq_mean_losses
                    finished_node_n += 1
                    buffer_pointer += 1
                    if len(token_sequence_buffer) == buffer_pointer:
                        token_sequence_buffer = np.concatenate((token_sequence_buffer, np.full(buffer_size, self.tokenizer.pad_id, dtype=np.int32)))
                        seq_mean_losses_buffer = np.concatenate((seq_mean_losses_buffer, np.full(buffer_size, float('+inf'), dtype=np.float32)))
            
            # Save finished node buffers to disk
            if buffer_pointer:
                np.save(buffer_file_token, token_sequence_buffer)
                np.save(buffer_file_loss, seq_mean_losses_buffer)

            # Update tree depth; separate nodes
            if tree_depth == self.max_tree_depth and working_nodes:
                # Find best working node
                best_mean_loss = working_nodes[0].token_mean_loss_sequence[-1]
                best_wirking_node = working_nodes[0]
                for node in working_nodes:
                    mean_loss = node.token_mean_loss_sequence[-1]
                    if mean_loss < best_mean_loss:
                        best_mean_loss = mean_loss
                        best_wirking_node = node

                # Update current root node
                current_root_node = best_wirking_node
                for _ in range(self.max_tree_depth - 1):
                    current_root_node = current_root_node.parent_node
                
                # Save current root node results to disk
                np.save(root_token_sequence_file, np.array(current_root_node.token_sequence, dtype=np.int32))
                np.save(root_mean_loss_sequence_file, np.array(current_root_node.token_mean_loss_sequence, dtype=np.float32))
                with open(root_tokens_file, 'w') as file:
                    json.dump(current_root_node.tokens, file)
                with open(root_vocab_file, 'w') as file:
                    json.dump(current_root_node.vocab, file)

                # Separate on winner (working) and looser nodes
                looser_node_n = 0
                winner_nodes: List[Node] = []
                for node in working_nodes:
                    parent_node = node
                    for _ in range(self.max_tree_depth - 1):
                        parent_node = parent_node.parent_node
                    if parent_node is current_root_node:
                        winner_nodes.append(node)
                    else:
                        looser_node_n += 1
                working_nodes = winner_nodes

                # Update tree depth
                tree_depth -= 1
            else:
                looser_node_n = 0

            # Update max sequence length and max batch size
            for node in working_nodes:
                if len(node.token_sequence) > self.max_seq_len:
                    # Update lengths
                    self.max_seq_len += 8
                    self.max_batch_size = self.len2batch_map[self.max_seq_len]

                    # Clear GPU memory
                    del self.model
                    gc.collect()
                    with self.device:
                        torch.cuda.empty_cache()

                    # Load model and compile
                    self.model = generate.load_model(self.model_path, self.model_variant, self.device)
                    self.model = torch.compile(self.model, mode='reduce-overhead', fullgraph=True)

            # Calc node statistics
            node_mult_factor = len(new_nodes) / prev_new_node_n
            prev_new_node_n = len(new_nodes)

            # Logging
            duration = time.monotonic() - start_time
            current_root_sequence_str = self.tokenizer.decode(current_root_node.token_sequence)
            print()
            print(f'{"Duration:".ljust(16)} {str(int(duration)).rjust(5)}')
            print(f'{"Tree depth:".ljust(16)} {str(tree_depth).rjust(5)}')
            print(f'{"Sequence len:".ljust(16)} {str(len(current_root_node.token_sequence)).rjust(5)}')
            print(f'{"Max seq len:".ljust(16)} {str(self.max_seq_len).rjust(5)}')
            print('Nodes:')
            print(f'{"Mult factor:".ljust(16)} {str(round(node_mult_factor, 2)).rjust(5)}')
            print(f'{"New:".ljust(16)} {str(len(new_nodes)).rjust(5)}')
            print(f'\t{"Working:".ljust(16)} {str(len(working_nodes)).rjust(5)}')
            print(f'\t{"Loosing:".ljust(16)} {str(looser_node_n).rjust(5)}')
            print(f'\t{"Finished:".ljust(16)} {str(finished_node_n).rjust(5)}')
            print(f'{"All finished:".ljust(16)} {str(buffer_pointer).rjust(5)}')
            print(f'Sequence: {current_root_sequence_str}')
            print(f'Sequence: {current_root_node.token_sequence}')
            print(f'Mean losses: {current_root_node.token_mean_loss_sequence}')
            logger.info(f'{"Duration:".ljust(16)} {str(int(duration)).rjust(5)}')
            logger.info(f'{"Tree depth:".ljust(16)} {str(tree_depth).rjust(5)}')
            logger.info(f'{"Sequence len:".ljust(16)} {str(len(current_root_node.token_sequence)).rjust(5)}')
            logger.info(f'{"Max seq len:".ljust(16)} {str(self.max_seq_len).rjust(5)}')
            logger.info('Nodes:')
            logger.info(f'{"Mult factor:".ljust(16)} {str(round(node_mult_factor, 2)).rjust(5)}')
            logger.info(f'{"New:".ljust(16)} {str(len(new_nodes)).rjust(5)}')
            logger.info(f'\t{"Working:".ljust(16)} {str(len(working_nodes)).rjust(5)}')
            logger.info(f'\t{"Loosing:".ljust(16)} {str(looser_node_n).rjust(5)}')
            logger.info(f'\t{"Finished:".ljust(16)} {str(finished_node_n).rjust(5)}')
            logger.info(f'{"All finished:".ljust(16)} {str(buffer_pointer).rjust(5)}')
            logger.info(f'Sequence: {current_root_sequence_str}')
            logger.info(f'Sequence: {current_root_node.token_sequence}')
            logger.info(f'Mean losses: {current_root_node.token_mean_loss_sequence}')

        # Find best finished node
        eos_mask = token_sequence_buffer == self.tokenizer.eos_id
        eos_mean_losses = seq_mean_losses_buffer[eos_mask]
        idx_0 = eos_mean_losses.argmin()
        idx_1 = eos_mask[idx_0].argmax()
        
        # Update result sequence
        result_token_sequence = token_sequence_buffer[idx_0, :idx_1+1]
        result_mean_loss_sequence = seq_mean_losses_buffer[idx_0, :idx_1+1]
        result_sequence_str = self.tokenizer.decode(result_token_sequence.tolist())

        # Logging
        total_duration = time.monotonic() - total_start_time
        print()
        print('Total:')
        print(f'\tDuration: {int(total_duration)}')
        print(f'\tSequence: {result_sequence_str}')
        print(f'\tSequence: {result_token_sequence}')
        print(f'\tMean losses: {result_mean_loss_sequence}')
        print(f'\tMean loss: {result_mean_loss_sequence[idx_1]}')
        logger.info('Total:')
        logger.info(f'\tDuration: {total_duration}')
        logger.info(f'\tSequence: {result_sequence_str}')
        logger.info(f'\tSequence: {result_token_sequence}')
        logger.info(f'\tMean losses: {result_mean_loss_sequence}')
        logger.info(f'\tMean loss: {result_mean_loss_sequence[idx_1]}')

        # Check result_sequence
        assert Counter(result_sequence_str.split()) == Counter(self.permuted_string.split())

        return result_sequence_str, math.exp(result_mean_loss_sequence[idx_1])


if __name__ == "__main__":
    variant = "9b"
    ckpt_path = f"/models/google/gemma-2-{variant}-pytorch/model.ckpt"
    tokenizer_path = f'/models/google/gemma-2-{variant}-pytorch/tokenizer.model'
    device: Optional[Literal['cpu', 'cuda']] = None
    seed = 12345
    max_tree_depth = 9
    top_p = 0.997

    # Log file name
    log_file_name = datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S')
    # log_file_name = '2025.01.20 13.01.16'

    # Initial string for root node
    initial_string = ""
    # initial_string = "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament"

    # Get input ids
    permuted_string = "sentence english normal a is this"
    # permuted_string = "this scrooge chimney chimney for the mistletoe"
    # permuted_string = "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge"
    # permuted_string = "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge walk give jump drive bake the sleep night laugh and"
    # permuted_string = "yuletide decorations gifts cheer holiday carol magi nutcracker polar grinch sleigh chimney workshop stocking ornament holly jingle beard naughty nice"
    # permuted_string = "yuletide decorations gifts cheer holiday carol magi nutcracker polar grinch sleigh chimney workshop stocking ornament holly jingle beard naughty nice sing cheer and of the is eat visit relax unwrap"
    # permuted_string = "hohoho candle poinsettia snowglobe peppermint eggnog fruitcake chocolate candy puzzle game doll toy workshop wonder believe dream hope peace joy merry season greeting card wrapping paper bow fireplace night cookie milk star wish wreath angel the to of and in that have it not with as you from we kaggle"
    # permuted_string = "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge walk give jump drive bake the sleep night laugh and yuletide decorations gifts cheer holiday carol magi nutcracker polar grinch sleigh chimney workshop stocking ornament holly jingle beard naughty nice sing cheer and of the is eat visit relax unwrap hohoho candle poinsettia snowglobe peppermint eggnog fruitcake chocolate candy puzzle game doll toy workshop wonder believe dream hope peace joy merry season greeting card wrapping paper bow fireplace night cookie milk star wish wreath angel the to of and in that have it not with as you from we kaggle"
    # permuted_string = "mistletoe scrooge yuletide nutcracker grinch unwrap hohoho poinsettia snowglobe eggnog fruitcake kaggle"

    # Config logger
    logging.basicConfig(filename=Path('./logs') / f'{log_file_name}.log', encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%Y.%m.%d %H:%M:%S')

    # Define device
    if device is None:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device in ['cpu', 'cuda']:
        torch_device = torch.device(device)
    else:
        raise ValueError('Wrong device argument')
    
    # Logging
    print(f'string: {permuted_string}')
    print(f'Model:  {variant}')
    print(f'Depth:  {max_tree_depth}')
    print(f'Top-p:  {top_p}')
    print(f'Device: {torch_device}')
    print()
    logger.info(f'string: {permuted_string}')
    logger.info(f'Model:  {variant}')
    logger.info(f'Depth:  {max_tree_depth}')
    logger.info(f'Top-p:  {top_p}')
    logger.info(f'Device: {torch_device}')

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate
    generator = Generator(
        initial_string,
        permuted_string,
        variant,
        ckpt_path,
        tokenizer_path,
        log_file_name,
        torch_device,
        max_tree_depth,
        top_p,
        )
    result_string, ppl = generator.generate_sequence()

    # Logging
    print()
    print(result_string)
    print(f'PPL: {ppl}')
    logger.info(result_string)
    logger.info(f'PPL: {ppl}')
