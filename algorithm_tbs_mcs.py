from collections import Counter, defaultdict
import dataclasses
import datetime
import gc
import json
import logging
import math
from pathlib import Path
import random
import statistics
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
    token_counter: Counter[int] = dataclasses.field(default_factory=Counter)
    vocab: List[int] = dataclasses.field(default_factory=list)
    sequence_mean_loss: float = float('+inf')


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


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
            mcs_n: int,
            ):
        self.permuted_string = " " + permuted_string
        self.model_variant = model_variant
        self.model_path = model_path
        self.log_file_name = log_file_name
        self.device = device
        self.mcs_n = mcs_n
        
        # Initialize initial string
        self.root_string_file = Path('./logs') / f'{self.log_file_name} root_string.json'
        assert not (self.root_string_file.is_file() and initial_string)
        if self.root_string_file.is_file():
            with open(self.root_string_file) as file:
                self.initial_string = json.load(file)
        else:
            self.initial_string = initial_string

        # Load tokenizer
        self.tokenizer = gemma_tokenizer.Tokenizer(tokenizer_path)

        # Initialize permuted tokens
        self.permuted_tokens = self.tokenizer.encode(self.permuted_string, bos=False, eos=False)
        self.permuted_token_counter = Counter(self.permuted_tokens)
        self.permuted_string_counter = Counter(self.permuted_string.split())

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
        self.max_seq_len = len(self.permuted_tokens) + 3  # BOS, EOS, extra
        self.max_seq_len = find_multiple(self.max_seq_len, 8)
        self.max_batch_size = self.len2batch_map[self.max_seq_len]

        # Load model and compile
        self.model = generate.load_model(self.model_path, self.model_variant, self.device)
        self.model = torch.compile(self.model, mode='reduce-overhead', fullgraph=True)

    def create_child_node(
            self,
            parent_node: Node,
            token: int,
            ) -> Node:
        # Form new token sequence
        token_sequence = parent_node.token_sequence.copy()
        token_sequence.append(token)

        # Update token sequence for non-single tokens
        tmp_token = token
        while (tmp_token in self.non_single_token_map and len(self.non_single_token_map[tmp_token]) == 1):
            tmp_token = self.non_single_token_map[tmp_token][0]
            token_sequence.append(tmp_token)

        # Check for BOS
        if parent_node.token_sequence == [self.tokenizer.bos_id]:  # check for BOS
            spaced_token_sequence = token_sequence
            token_sequence_str = self.tokenizer.decode(token_sequence)[1:]  # drop first space
            token_sequence = self.tokenizer.encode(token_sequence_str, bos=True, eos=False)
        else:
            token_sequence_str = f" {self.tokenizer.decode(token_sequence)}"  # add first space
            spaced_token_sequence = self.tokenizer.encode(token_sequence_str, bos=True, eos=False)

        # Form token counter and vocab
        token_counter = self.permuted_token_counter - Counter(spaced_token_sequence)
        vocab = list(set(token_counter.keys()) & set(self.spaced_tokens))

        # Create and return child node
        return Node(
            parent_node=parent_node,
            token_sequence=token_sequence,
            token_counter=token_counter,
            vocab=vocab,
            )

    def get_seq_mean_losses(self,
                             token_sequence_batch: List[List[int]],
                             vocab_batch: List[List[int]]) -> List[float]:
        # Generate top tokens
        print('Batch...')
        logger.info('Batch...')
        (top_tokens_batch, top_tokens_mean_losses_batch,
         seq_mean_losses_batch, seq_mean_loss_batch) = generate.generate(
            token_sequence_batch,
            vocab_batch,
            self.model, 
            self.device,
            self.tokenizer.pad_id,
            self.max_batch_size,
            self.max_seq_len,
            self.max_vocab_len,
            1.1,
            )

        return seq_mean_loss_batch

    def generate_random_sequence(
            self,
            node_token_sequence: List[int],
            spaced_token_counter: Counter[int],
            n: int,
            ):
        for _ in range(n):
            # Generate random token sequences
            token_sequence = node_token_sequence.copy()
            spaced_tokens = list(spaced_token_counter.elements())
            while spaced_tokens:
                token = random.choice(spaced_tokens)
                spaced_tokens.remove(token)
                token_sequence.append(token)
                while (token in self.non_single_token_map and len(self.non_single_token_map[token]) == 1):
                    token = self.non_single_token_map[token][0]
                    token_sequence.append(token)
            token_sequence.append(self.tokenizer.eos_id)

            # Check sequence
            sequence_str = self.tokenizer.decode(token_sequence)
            assert Counter(sequence_str.split()) == self.permuted_string_counter

            yield token_sequence

    def get_next_root_node(self, root_node: Node) -> Node:
        # Get child nodes
        child_nodes = [self.create_child_node(root_node, token) for token in root_node.vocab]

        # Get average loss for all child nodes
        child_node_losses = []
        for node in child_nodes:
            # Get counter for spaced tokens
            spaced_token_counter = node.token_counter.copy()
            for token in node.token_counter:
                if token not in self.spaced_tokens:
                    del spaced_token_counter[token]

            # Get losses for all sequences (batching)
            token_sequence_batch = []
            vocab_batch = []
            seq_mean_losses = []
            for token_sequence in self.generate_random_sequence(node.token_sequence, spaced_token_counter, self.mcs_n):
                token_sequence_batch.append(token_sequence)
                vocab_batch.append(node.vocab)
                if len(token_sequence_batch) == self.max_batch_size:
                    seq_mean_loss_batch = self.get_seq_mean_losses(token_sequence_batch, vocab_batch)
                    seq_mean_losses.extend(seq_mean_loss_batch)
                    token_sequence_batch = []
            if token_sequence_batch:
                seq_mean_loss_batch = self.get_seq_mean_losses(token_sequence_batch, vocab_batch)
                seq_mean_losses.extend(seq_mean_loss_batch)

            # Calc average loss for child node sequences
            child_node_loss = statistics.mean(seq_mean_losses)
            child_node_losses.append(child_node_loss)
            node.sequence_mean_loss = child_node_loss

        # Pick-up child node with the best avg loss
        return child_nodes[child_node_losses.index(min(child_node_losses))]

    def generate_sequence(self) -> Tuple[str, float]:
        # Initialize current root node
        if self.initial_string != "":
            token_sequence = self.tokenizer.encode(self.initial_string, bos=True, eos=False)
            spaced_token_sequence = self.tokenizer.encode(f" {self.initial_string}", bos=True, eos=False)
        else:
            token_sequence = [self.tokenizer.bos_id]
            spaced_token_sequence = token_sequence
        token_counter = self.permuted_token_counter - Counter(spaced_token_sequence)
        vocab = list(set(token_counter.keys()) & set(self.spaced_tokens))
        current_root_node = Node(
            token_sequence=token_sequence,
            vocab=vocab,
            )

        current_root_sequence_str = self.tokenizer.decode(current_root_node.token_sequence)
        total_start_time = time.monotonic()
        while current_root_node.vocab:
            start_time = time.monotonic()
            current_root_node = self.get_next_root_node(current_root_node)
            
            # Save current root node results to disk
            current_root_sequence_str = self.tokenizer.decode(current_root_node.token_sequence)
            with open(self.root_string_file, 'w') as file:
                json.dump(current_root_sequence_str, file)

            # Logging
            duration = time.monotonic() - start_time
            print()
            print(f'{"Duration:".ljust(16)} {str(int(duration)).rjust(5)}')
            print(f'{"Vocab len:".ljust(16)} {str(len(current_root_node.vocab)).rjust(5)}')
            print(f'{"Sequence len:".ljust(16)} {str(len(current_root_node.token_sequence)).rjust(5)}')
            print(f'{"Max seq len:".ljust(16)} {str(self.max_seq_len).rjust(5)}')
            print(f'Sequence: {current_root_sequence_str}')
            print(f'Sequence: {current_root_node.token_sequence}')
            print(f'Mean loss: {current_root_node.sequence_mean_loss}')
            logger.info(f'{"Duration:".ljust(16)} {str(int(duration)).rjust(5)}')
            logger.info(f'{"Vocab len:".ljust(16)} {str(len(current_root_node.vocab)).rjust(5)}')
            logger.info(f'{"Sequence len:".ljust(16)} {str(len(current_root_node.token_sequence)).rjust(5)}')
            logger.info(f'{"Max seq len:".ljust(16)} {str(self.max_seq_len).rjust(5)}')
            logger.info(f'Sequence: {current_root_sequence_str}')
            logger.info(f'Sequence: {current_root_node.token_sequence}')
            logger.info(f'Mean loss: {current_root_node.sequence_mean_loss}')

        # Logging
        total_duration = time.monotonic() - total_start_time
        print()
        print('Total:')
        print(f'\tDuration: {int(total_duration)}')
        print(f'\tSequence: {current_root_sequence_str}')
        print(f'\tSequence: {current_root_node.token_sequence}')
        print(f'\tMean loss: {current_root_node.sequence_mean_loss}')
        logger.info('Total:')
        logger.info(f'\tDuration: {int(total_duration)}')
        logger.info(f'\tSequence: {current_root_sequence_str}')
        logger.info(f'\tSequence: {current_root_node.token_sequence}')
        logger.info(f'\tMean loss: {current_root_node.sequence_mean_loss}')

        # Check result_sequence
        assert Counter(current_root_sequence_str.split()) == self.permuted_string_counter

        return current_root_sequence_str, math.exp(current_root_node.sequence_mean_loss)


if __name__ == "__main__":
    variant = "9b"
    ckpt_path = f"./models/google/gemma-2-{variant}-pytorch/model.ckpt"
    tokenizer_path = f'./models/google/gemma-2-{variant}-pytorch/tokenizer.model'
    device: Optional[Literal['cpu', 'cuda']] = None
    seed = 12345
    mcs_n = 100_000  # Monte-Carlo search number

    # Log file name
    log_file_name = datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S')
    # log_file_name = '2025.01.20 13.01.16'

    # Initial string for root node
    # initial_string = ""
    initial_string = "reindeer mistletoe"

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
    print(f'Device: {torch_device}')
    print(f'MCS number: {mcs_n}')
    print()
    logger.info(f'string: {permuted_string}')
    logger.info(f'Model:  {variant}')
    logger.info(f'Device: {torch_device}')
    logger.info(f'MCS number: {mcs_n}')

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
        mcs_n,
        )
    result_string, ppl = generator.generate_sequence()

    # Logging
    print()
    print(result_string)
    print(f'PPL: {ppl}')
    logger.info(result_string)
    logger.info(f'PPL: {ppl}')
