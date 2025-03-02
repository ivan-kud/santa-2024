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
    token_sequence: List[int] = dataclasses.field(default_factory=list)
    vocab: List[int] = dataclasses.field(default_factory=list)
    sequence_mean_loss: float = float('+inf')
    before_finished: bool = False
    finished: bool = False


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
            ):
        self.permuted_string = " " + permuted_string
        self.model_variant = model_variant
        self.model_path = model_path
        self.log_file_name = log_file_name
        self.device = device
        self.max_tree_depth = max_tree_depth
        
        # Initialize initial string
        self.root_string_file = Path('./logs') / f'{self.log_file_name} root_string.json'
        assert not (self.root_string_file.is_file() and initial_string)
        if self.root_string_file.is_file():
            with open(self.root_string_file) as file:
                self.initial_string = json.load(file)
        else:
            self.initial_string = " " + initial_string

        # Load tokenizer
        self.tokenizer = gemma_tokenizer.Tokenizer(tokenizer_path)

        # Initialize permuted tokens
        self.permuted_tokens = self.tokenizer.encode(self.permuted_string, bos=False, eos=False)
        self.permuted_token_counter = Counter(self.permuted_tokens)

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

    def update_token_sequence(
            self,
            prev_token_sequence: List[int],
            new_token: int,
            ) -> Tuple[List[int], List[int]]:
        # Form new token sequence
        new_token_sequence = [self.tokenizer.bos_id, new_token]

        # Update token sequence for non-single tokens
        token = new_token
        while (token in self.non_single_token_map and len(self.non_single_token_map[token]) == 1):
            token = self.non_single_token_map[token][0]
            new_token_sequence.append(token)

        # Form tokens and vocab
        new_token_sequence.extend(prev_token_sequence[1:])
        token_counter = self.permuted_token_counter - Counter(new_token_sequence)
        vocab = list(set(token_counter.keys()) & set(self.spaced_tokens))

        return new_token_sequence, vocab

    def create_child_node(
            self,
            parent_node: Node,
            token: int,
            token_mean_loss: float,
            seq_mean_losses: List[float],
            seq_mean_loss: float,
            ) -> Node:
        token_sequence, vocab = self.update_token_sequence(parent_node.token_sequence, token)

        # Check for last token
        while len(vocab) == 1 and not parent_node.before_finished:
            last_token = vocab[0]
            token_sequence, vocab = self.update_token_sequence(token_sequence, last_token)

        # Check for no tokens
        if not vocab and not parent_node.before_finished:
            token_sequence_str = self.tokenizer.decode(token_sequence)[1:]  # exept first space
            token_sequence = self.tokenizer.encode(token_sequence_str, bos=True, eos=True)
            vocab = [self.tokenizer.bos_id]
            before_finished = True
        else:
            before_finished = False

        # Check for finished
        if parent_node.before_finished:
            token_sequence = token_sequence[1:]
            vocab = []
            finished = True
        else:
            finished = False

        # Create and return child node
        return Node(
            parent_node=parent_node,
            token_sequence=token_sequence,
            vocab=vocab,
            sequence_mean_loss=seq_mean_loss,
            before_finished=before_finished,
            finished=finished,
            )

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
            1.1,
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
            sequence_mean_loss_buffer = np.load(buffer_file_loss)
            buffer_pointer = np.sum(token_sequence_buffer == self.tokenizer.pad_id, axis=1).argmax()
        else:
            # Initialize buffers for finished nodes
            token_sequence_buffer = np.full(buffer_size, self.tokenizer.pad_id, dtype=np.int32)
            sequence_mean_loss_buffer = np.full(buffer_size[0], float('+inf'), dtype=np.float32)        
            buffer_pointer = 0

        # Initialize current root node
        if self.initial_string != " ":
            token_sequence = self.tokenizer.encode(self.initial_string, bos=True, eos=True)
        else:
            token_sequence = [self.tokenizer.bos_id, self.tokenizer.eos_id]
        token_counter = self.permuted_token_counter - Counter(token_sequence)
        vocab = list(set(token_counter.keys()) & set(self.spaced_tokens))
        current_root_node = Node(
            token_sequence=token_sequence,
            vocab=vocab,
            )

        tree_depth = 0
        working_nodes = [current_root_node]
        prev_new_node_n = len(working_nodes)
        current_root_sequence_str = self.tokenizer.decode(current_root_node.token_sequence)
        total_start_time = time.monotonic()
        while working_nodes:
            start_time = time.monotonic()
            new_nodes = self.get_child_nodes(working_nodes)
            tree_depth += 1

            # Separate new nodes on working and finished nodes
            finished_node_n = 0
            working_nodes: List[Node] = []
            for node in new_nodes:
                if node.finished:
                    # Store results of finished nodes to buffers
                    token_sequence_buffer[buffer_pointer, :len(node.token_sequence)] = node.token_sequence
                    sequence_mean_loss_buffer[buffer_pointer] = node.sequence_mean_loss
                    finished_node_n += 1
                    buffer_pointer += 1
                    if len(token_sequence_buffer) == buffer_pointer:
                        token_sequence_buffer = np.concatenate((token_sequence_buffer, np.full(buffer_size, self.tokenizer.pad_id, dtype=np.int32)))
                        sequence_mean_loss_buffer = np.concatenate((sequence_mean_loss_buffer, np.full(buffer_size[0], float('+inf'), dtype=np.float32)))
                else:
                    working_nodes.append(node)
            
            # Save finished node buffers to disk
            if buffer_pointer:
                np.save(buffer_file_token, token_sequence_buffer)
                np.save(buffer_file_loss, sequence_mean_loss_buffer)

            # Update tree depth; separate nodes
            if tree_depth == self.max_tree_depth and working_nodes:
                # Find best working node
                best_mean_loss = working_nodes[0].sequence_mean_loss
                best_wirking_node = working_nodes[0]
                for node in working_nodes:
                    if node.sequence_mean_loss < best_mean_loss:
                        best_mean_loss = node.sequence_mean_loss
                        best_wirking_node = node

                # Update current root node
                current_root_node = best_wirking_node
                for _ in range(self.max_tree_depth - 1):
                    current_root_node = current_root_node.parent_node
                
                # Save current root node results to disk
                current_root_sequence_str = self.tokenizer.decode(current_root_node.token_sequence)
                with open(self.root_string_file, 'w') as file:
                    json.dump(current_root_sequence_str, file)

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
            print()
            print(f'{"Duration:".ljust(16)} {str(int(duration)).rjust(5)}')
            print(f'{"Tree depth:".ljust(16)} {str(tree_depth).rjust(5)}')
            print(f'{"Sequence len:".ljust(16)} {str(len(current_root_node.token_sequence)).rjust(5)}')
            print(f'{"Max seq len:".ljust(16)} {str(self.max_seq_len).rjust(5)}')
            print('Nodes:')
            print(f'\t{"All finished:".ljust(16)} {str(buffer_pointer).rjust(5)}')
            print(f'\t{"Mult factor:".ljust(16)} {str(round(node_mult_factor, 2)).rjust(5)}')
            print(f'\t{"New:".ljust(16)} {str(len(new_nodes)).rjust(5)}')
            print(f'\t\t{"Working:".ljust(16)} {str(len(working_nodes)).rjust(5)}')
            print(f'\t\t{"Loosing:".ljust(16)} {str(looser_node_n).rjust(5)}')
            print(f'\t\t{"Finished:".ljust(16)} {str(finished_node_n).rjust(5)}')
            print(f'Sequence: {current_root_sequence_str}')
            print(f'Sequence: {current_root_node.token_sequence}')
            print(f'Mean loss: {current_root_node.sequence_mean_loss}')
            logger.info(f'{"Duration:".ljust(16)} {str(int(duration)).rjust(5)}')
            logger.info(f'{"Tree depth:".ljust(16)} {str(tree_depth).rjust(5)}')
            logger.info(f'{"Sequence len:".ljust(16)} {str(len(current_root_node.token_sequence)).rjust(5)}')
            logger.info(f'{"Max seq len:".ljust(16)} {str(self.max_seq_len).rjust(5)}')
            logger.info('Nodes:')
            logger.info(f'\t{"All finished:".ljust(16)} {str(buffer_pointer).rjust(5)}')
            logger.info(f'\t{"Mult factor:".ljust(16)} {str(round(node_mult_factor, 2)).rjust(5)}')
            logger.info(f'\t{"New:".ljust(16)} {str(len(new_nodes)).rjust(5)}')
            logger.info(f'\t\t{"Working:".ljust(16)} {str(len(working_nodes)).rjust(5)}')
            logger.info(f'\t\t{"Loosing:".ljust(16)} {str(looser_node_n).rjust(5)}')
            logger.info(f'\t\t{"Finished:".ljust(16)} {str(finished_node_n).rjust(5)}')
            logger.info(f'Sequence: {current_root_sequence_str}')
            logger.info(f'Sequence: {current_root_node.token_sequence}')
            logger.info(f'Mean loss: {current_root_node.sequence_mean_loss}')

        # Find best finished node
        best_loss_index = sequence_mean_loss_buffer.argmin()
        eos_mask = token_sequence_buffer == self.tokenizer.eos_id
        eos_index = eos_mask[best_loss_index].argmax()
        
        # Update result sequence
        result_token_sequence = token_sequence_buffer[best_loss_index, :eos_index+1]
        result_mean_loss_sequence = sequence_mean_loss_buffer[best_loss_index]
        result_sequence_str = self.tokenizer.decode(result_token_sequence.tolist())

        # Logging
        total_duration = time.monotonic() - total_start_time
        print()
        print('Total:')
        print(f'\tDuration: {int(total_duration)}')
        print(f'\tSequence: {result_sequence_str}')
        print(f'\tSequence: {result_token_sequence}')
        print(f'\tMean loss: {result_mean_loss_sequence}')
        logger.info('Total:')
        logger.info(f'\tDuration: {int(total_duration)}')
        logger.info(f'\tSequence: {result_sequence_str}')
        logger.info(f'\tSequence: {result_token_sequence}')
        logger.info(f'\tMean loss: {result_mean_loss_sequence}')

        # Check result_sequence
        assert Counter(result_sequence_str.split()) == Counter(self.permuted_string.split())

        return result_sequence_str, math.exp(result_mean_loss_sequence)


if __name__ == "__main__":
    variant = "9b"
    ckpt_path = f"/models/google/gemma-2-{variant}-pytorch/model.ckpt"
    tokenizer_path = f'/models/google/gemma-2-{variant}-pytorch/tokenizer.model'
    device: Optional[Literal['cpu', 'cuda']] = None
    seed = 12345
    max_tree_depth = 6

    # Log file name
    log_file_name = datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S')
    # log_file_name = '2025.01.20 13.01.16'

    # Initial string for root node
    initial_string = ""
    # initial_string = "gingerbread bake the chimney family advent night elf and mistletoe reindeer ornament"

    # Get input ids
    # permuted_string = "sentence english normal a is this"
    # permuted_string = "this scrooge chimney chimney for the mistletoe"
    # permuted_string = "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge"
    permuted_string = "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge walk give jump drive bake the sleep night laugh and"
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
    print(f'Device: {torch_device}')
    print()
    logger.info(f'string: {permuted_string}')
    logger.info(f'Model:  {variant}')
    logger.info(f'Depth:  {max_tree_depth}')
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
        )
    result_string, ppl = generator.generate_sequence()

    # Logging
    print()
    print(result_string)
    print(f'PPL: {ppl}')
    logger.info(result_string)
    logger.info(f'PPL: {ppl}')
