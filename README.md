# Santa 2024 - The Perplexity Permutation Puzzle

## Task

Achieve the lowest perplexity score by rearranging words in the text.

## Modifications in Gemma-2 model

1. Removed the sliding attention window since we only have short texts.
2. Added calculation of top-p directly in the forward method.
3. The code is prepared for compilation using torch.compile without graph breaks.

## Implemented algorithms

1. Tree-based search approach to explore possible permutations of the words.
2. Tree-based search in reverse order: from the end of the text to the beginning.
3. Tree-based search with Monte Carlo sampling.

## Repository structure

```raw
.
├── results
│   └── results.txt             # Results
├── algorithm_tbs_mcs.py        # Tree-based search with Monte Carlo sampling
├── algorithm_tbs.py            # Tree-based search
├── algorithm_tbs_reverse.py    # Tree-based search in reverse order
├── config.py                   # Gemma-2 configuration
├── generate.py                 # Gemma-2 generation
├── model.py                    # Gemma-2 model
├── tokenizer.py                # Tokenizer
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
