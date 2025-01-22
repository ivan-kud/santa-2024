from collections import Counter
import contextlib
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Optional, Literal, Tuple, List

import numpy as np
import torch

import config
import model as gemma_model
import tokenizer as gemma_tokenizer


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def load_model(
        ckpt_path: str,
        variant: str,
        device: torch.device,
        ) -> gemma_model.GemmaForCausalLM:
    quant = False
    cuda_dtype = "float16"

    print("Loading model...")
    start = time.monotonic()
    model_config = config.get_model_config(variant)
    model_config.dtype = "float32" if device.type == "cpu" else cuda_dtype
    model_config.quant = quant
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_state_dict(
            torch.load(ckpt_path, mmap=True, weights_only=True,)['model_state_dict'],
            strict=False,
        )
        model = model.to(device).eval()
    print(f"Done: in {time.monotonic() - start}")

    return model


def prepare_inputs(
    input_ids_list: List[List[int]],
    vocab_ids_list: List[List[int]],
    pad_id: int,
    device: torch.device,
    max_batch_size: int,
    max_seq_len: int,
    max_vocab_len: int,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, L], [B, Vc], [1], [1]
    assert max_batch_size >= len(input_ids_list)
    input_ids = torch.full((max_batch_size, max_seq_len), pad_id, dtype=torch.long, device=device)
    vocab_ids = torch.full((max_batch_size, max_vocab_len), pad_id, dtype=torch.long, device=device)
    for i, i_ids, v_ids in zip(range(max_batch_size), input_ids_list, vocab_ids_list):
        assert max_seq_len >= len(i_ids)
        assert max_vocab_len >= len(v_ids)
        input_ids[i, :len(i_ids)] = torch.tensor(i_ids, dtype=torch.long, device=device)
        vocab_ids[i, :len(v_ids)] = torch.tensor(v_ids, dtype=torch.long, device=device)

    pad_id_tensor = torch.tensor([pad_id], dtype=torch.long, device=device)
    top_p_tensor = torch.tensor([top_p], dtype=torch.float, device=device)

    return input_ids.contiguous(), vocab_ids.contiguous(), pad_id_tensor, top_p_tensor  # [B, L], [B, Vc], [1], [1]


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


@torch.no_grad()
def generate(
    input_ids_list: List[List[int]],
    vocab_ids_list: List[List[int]],
    model: gemma_model.GemmaForCausalLM,
    device: torch.device,
    pad_id: int,
    max_batch_size: int,
    max_seq_len: int,
    max_vocab_len: int,
    top_p: float,
) -> Tuple[List[List[int]], List[List[int]], List[List[float]], List[float]]:
    current_batch_size = len(input_ids_list)
    max_seq_len = find_multiple(max_seq_len, 8)
    max_vocab_len = find_multiple(max_vocab_len, 8)

    input_ids, vocab_ids, pad_id_tensor, top_p_tensor = prepare_inputs(
        input_ids_list,
        vocab_ids_list,
        pad_id,
        device,
        max_batch_size,
        max_seq_len,
        max_vocab_len,
        top_p,
        )  # [B, L], [B, Vc]

    top_token_ids, top_token_mean_losses, mean_losses = model(
        input_ids,
        vocab_ids,
        pad_id_tensor,
        top_p_tensor,
        )  # [B, L, Vc], [B, L, Vc], [B, L]

    # Get last non-pad indices
    last_non_pad_indices = []
    for b in range(current_batch_size):
        sequence_id = (input_ids[b, :].tolist() + [pad_id]).index(pad_id) - 1
        last_non_pad_indices.append(sequence_id)
    last_non_pad_indices = torch.tensor(last_non_pad_indices, dtype=torch.long, device=device)  # [Bc]
    batch_indices = torch.arange(current_batch_size, dtype=torch.long, device=device)  # [Bc]

    # Get output for last non-pad token in the sequences
    top_seq_token_ids = top_token_ids[batch_indices, last_non_pad_indices, :].tolist()  # [Bc, Vc]
    top_seq_token_mean_losses = top_token_mean_losses[batch_indices, last_non_pad_indices, :].tolist()  # [Bc, Vc]
    mean_seq_losses = mean_losses[batch_indices, last_non_pad_indices].tolist()  # [Bc]

    return (top_seq_token_ids,  # [Bc, Vc]
            top_seq_token_mean_losses,  # [Bc, Vc]
            mean_losses[:current_batch_size, :].tolist(),  # [Bc, L]
            mean_seq_losses)  # [Bc]


if __name__ == "__main__":
    # Arguments
    variant = "9b"
    ckpt_path = f"/models/google/gemma-2-{variant}-pytorch/model.ckpt"
    tokenizer_path = f'/models/google/gemma-2-{variant}-pytorch/tokenizer.model'
    device: Optional[Literal['cpu', 'cuda']] = None
    seed = 12345
    max_batch_size = 16
    max_seq_len = 32
    max_vocab_len = 32
    top_p = 0.95

    # Define device
    if device is None:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device in ['cpu', 'cuda']:
        torch_device = torch.device(device)
    else:
        raise ValueError('Wrong device argument')
    print('Using device:', torch_device)

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load tokenizer
    tokenizer = gemma_tokenizer.Tokenizer(tokenizer_path)

    # Get input ids
    # input_ids_list = [
    #     [2, 13060, 26299, 4500,   476,   603,  736, 1],
    #     [2, 27894,   573, 1163, 36271, 25341, 8426, 4320, 573, 5929, 1],
    #     ]
    permutation_strings = [
        "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament",
        "reindeer and elf night sleep walk gingerbread bake family laugh ornament give advent fireplace chimney drive jump the scrooge mistletoe",
        "reindeer and elf night sleep walk gingerbread bake family laugh the fireplace chimney jump give advent scrooge mistletoe ornament drive",
        "reindeer and elf family night sleep walk the gingerbread drive laugh jump bake give advent scrooge chimney fireplace mistletoe ornament",
        ]
    input_ids_list = [tokenizer.encode(prompt, bos=True, eos=True) for prompt in permutation_strings]

    # Init vocab
    current_tokens_list = []
    vocab_ids_list = []
    for input_ids in input_ids_list:
        current_tokens = dict(Counter(input_ids))
        current_tokens[tokenizer.bos_id] -= 1
        if current_tokens[tokenizer.bos_id] == 0:
            del current_tokens[tokenizer.bos_id]
        current_tokens_list.append(current_tokens)
        vocab_ids_list.append(list(current_tokens.keys()))

    # Load model
    model = load_model(ckpt_path, variant, torch_device)

    # # Compile
    # print('\n---\nCompiling...')
    # model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

    # Generate
    print('\n---\nGeneration...')
    top_token_ids, top_token_mean_losses, mean_losses, mean_seq_losses = generate(
        input_ids_list,
        vocab_ids_list,
        model, 
        torch_device,
        tokenizer.pad_id,
        max_batch_size,
        max_seq_len,
        max_vocab_len,
        top_p,
        )

    # Output
    ppl_list = []
    for i in range(len(input_ids_list)):
        ppl = math.exp(mean_seq_losses[i])
        ppl_list.append(ppl)
        print(f"\nInput:\n{tokenizer.decode(input_ids_list[i])}\n{input_ids_list[i]}")
        print(f"\nVocab:\n{tokenizer.decode(vocab_ids_list[i])}\n{vocab_ids_list[i]}")
        print(f"\nTop-p:\n{tokenizer.decode(top_token_ids[i])}\n{top_token_ids[i]}")
        print(f"\nTop-p average losses:\n{top_token_mean_losses[i]}")
        print(f"\nSequence average losses:\n{mean_losses[i]}")
        print(f"\nWhole sequence average loss: {mean_seq_losses[i]}")
        print(f"Whole sequence PPL: {ppl:.2f}")
    print("\nAverage PPL:", statistics.mean(ppl_list))

    # # Profiling
    # profiling_dir = Path('./profiling')
    # os.makedirs(profiling_dir, exist_ok=True)
    # max_seq_len = find_multiple(max_seq_len, 8)
    # max_vocab_len = find_multiple(max_vocab_len, 8)
    # input_ids, vocab_ids, pad_id_tensor, top_p_tensor = prepare_inputs(
    #     input_ids_list,
    #     vocab_ids_list,
    #     tokenizer.pad_id,
    #     torch_device,
    #     max_batch_size,
    #     max_seq_len,
    #     max_vocab_len,
    #     top_p,
    #     )
    # with torch.no_grad():
        # # Check graph breaks
        # print('\n---\nExplanation of graph breaks...')
        # explanation = torch._dynamo.explain(model)(input_ids, vocab_ids, pad_id_tensor, top_p_tensor)
        # with open(profiling_dir / 'explain.txt', 'w') as f:
        #     f.write(str(explanation))
        # if torch_device.type == 'cuda': torch.cuda.synchronize(torch_device)

        # # Export graph (for graphs without breaks only)
        # print('\n---\nExporting of graph...')
        # exported_model = torch.export.export(model, args=(input_ids, vocab_ids, pad_id_tensor, top_p_tensor))
        # with open(profiling_dir / 'export.txt', 'w') as f:
        #     f.write(str(exported_model))
        # if torch_device.type == 'cuda': torch.cuda.synchronize(torch_device)

        # # Compile model with checking graph breaks and recompilations
        # print('\n---\nCompiling with profiling...')
        # with torch._dynamo.utils.CompileProfiler() as prof:
        #     encoder = torch.compile(model, fullgraph=True, backend=prof)
        # with open(profiling_dir / 'compile.txt', 'w') as f:
        #     f.write(str(prof.report()))
        # if torch_device.type == 'cuda': torch.cuda.synchronize(torch_device)

        # # Record memory history
        # print('\n---\nRecording memory history...')
        # print('\tcompiling...')
        # model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
        # print('\trecording...')
        # torch.cuda.memory._record_memory_history()
        # model(input_ids)
        # torch.cuda.memory._dump_snapshot(str(profiling_dir / "memory_record.pickle"))
        # if torch_device.type == 'cuda': torch.cuda.synchronize(torch_device)

        # # Compile
        # print('\n---\nCompiling...')
        # model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
        # if torch_device.type == 'cuda': torch.cuda.synchronize(torch_device)

        # # Warm up
        # print('\n---\nWarming up...')
        # for i in range(5):
        #     print(f'\t\t{i=}')
        #     model(input_ids, vocab_ids, pad_id_tensor, top_p_tensor)
        # if torch_device.type == 'cuda': torch.cuda.synchronize(torch_device)

        # # Profile
        # print('\n---\nProfiling...')
        # with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True) as prof:
        #     for _ in range(5):
        #         model(input_ids, vocab_ids, pad_id_tensor, top_p_tensor)
        # with open(profiling_dir / 'profile.txt', 'w') as f:
        #     f.write(str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)))
        # prof.export_chrome_trace(str(profiling_dir / 'profile_trace.json'))
        # prof.export_memory_timeline(str(profiling_dir / 'profile_memory.html'))
        # if torch_device.type == 'cuda': torch.cuda.synchronize(torch_device)

        # # Generate
        # print('\n---\nGeneration...')
        # for _ in range(3):
        #     top_token_ids, top_token_mean_losses, mean_losses, mean_seq_losses = generate(
        #         input_ids_list,
        #         vocab_ids_list,
        #         model, 
        #         torch_device,
        #         tokenizer.pad_id,
        #         max_batch_size,
        #         max_seq_len,
        #         max_vocab_len,
        #         top_p,
        #         )
