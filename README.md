# Isnad Detection

Code for training and applying a CRF tagger that marks isnads in
[OpenITI](https://openiti.org/) Arabic texts.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

Run every script from the repository root, because each one reads
`isnadTaggedLocations.json` from the current directory. That file records which
line ranges of each book have been annotated by hand.

## Tagging new texts with a pretrained model

Two trained models are included: `tokenIsnadModel_28-7-20.crf` (newer) and
`tokenIsnadModel_27-11-19.crf`.

```sh
# Tag a single OpenITI file or a folder of them
uv run python tagIsnads.py <text_or_folder> tokenIsnadModel_28-7-20.crf tagged.json

# Write the predicted tags back into the texts as @ISB@ / @ISE@ markers
uv run python createEvalData.py <text_or_folder> tagged.json <output_file_or_folder>
```

`tagged.json` has one JSON object per paragraph, holding its tokens and
BIO tags (`B_Isnad`, `I_Isnad`, `O`). Pass `--headless_sequences` to
`createEvalData.py` to also mark isnads whose predicted span lacks a `B` tag.

## Training a model

1. **Build the training data** from a folder of OpenITI texts annotated with
   `@Isnad_Beg@` / `@Isnad_End@` markers:

   ```sh
   uv run python createTrainingData.py <annotated_texts_folder> trainingData.json
   ```

   `trainingDataChunks_7-28-20.json` is a ready-made copy of this output.

2. **Train the CRF:**

   ```sh
   uv run python trainCRF.py --features tokens trainingData.json model.crf
   ```

   Useful options:

   - `--sampleCount N --chunkSize 200`: split each annotated section into
     200-token chunks at N different random offsets, and train on the chunks.
   - `--windowSize 5`: how many tokens on each side of a token are used as
     features.
   - `--crossval`: run leave-one-out cross-validation and write per-fold scores
     to the output file instead of saving a model. This is slow on large data.
   - `--crossvalPredict`: run 10-fold cross-validation and write each held-out
     prediction to the output file.
   - `--features vectors`: use word embeddings instead of tokens. This needs a
     GloVe-format `vectors_priOnly.txt` in the repository root, which is not
     included.

## Other scripts

- `removeUnverifiedTags.py` copies `taggedTexts/` to `taggedTexts_verifiedOnly/`,
  keeping isnad markers (renamed to `@ISB@` / `@ISE@`) inside the
  hand-verified ranges listed in `isnadTaggedLocations.json` and removing all
  `@...@` tags elsewhere. Both folder names are hard-coded.

## License

MIT. See [LICENSE](LICENSE).
