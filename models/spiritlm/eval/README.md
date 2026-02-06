# STSP Evaluation
The Speech-Text Sentiment Preservation (STSP) benchmark is made of a collection of speech and text prompts in the positive, negative or neutral sentiment.
Given a spoken or written prompt , the task consists in generating a text or speech sequence of tokens that preserves the sentiment of the prompt.

The sentiment of the prompt is evaluated automatically with a sentiment/emotion classifier in speech or text depending of the output modality.
Based on these, we derive a STSP accuracy score.

## Data Download
Download the data as well as the speech/text classifier checkpoints via this [link](https://dl.fbaipublicfiles.com/textless_nlp/spiritlm/stsp.tar.gz)
then extract the data into the folder `{spiritlm ROOT FOLDER}/data/stsp_data`
```
cd {spiritlm ROOT FOLDER}
mkdir data/stsp_data
tar -xvzf stsp.tar.gz -C data/stsp_data --strip-components=1
```
Run the following script to check the dataset is all correctly present:
```
python spiritlm/eval/stsp/sanity_check_download.py
```
## Data structure
The dataset contains 3 folders:
- `data`: raw audio files
- `manifest`: data splits
- `model`: speech/text classifier checkpoints
### Data
The raw audio files for
- `emov`: EMOV
- `expresso/conversational`: EXPRESSO-ASR
- `expresso/read`: EXPRESSO-READ

### Manifest
The train/validation/test splits, concretely we have:

#### EMOV
- 1053 records for emov train split at `manifest/emov/emov.train.jsonl`
- 351 records for emov dev split at `manifest/emov/emov.dev.jsonl`
- 351 records for emov test split at `manifest/emov/emov.test.jsonl`

#### EXPRESSO-ASR
- 1373 records for EXPRESSO-ASR train split at `manifest/expresso/expresso_asr.train`
- 479 records for EXPRESSO-ASR dev at `manifest/expresso/expresso_asr.dev.jsonl`
- 462 records for EXPRESSO-ASR test split at `manifest/expresso/expresso_asr.test.jsonl`

#### EXPRESSO-READ
- 1024 records for EXPRESSO-READ train split at `manifest/expresso/expresso_read.train`
- 60 records for EXPRESSO-READ dev at `manifest/expresso/expresso_read.dev.jsonl`
- 54 records for EXPRESSO-READ test split at `manifest/expresso/expresso_read.test.jsonl`

#### Few-shot Samples
The subset from EXPRESSO-ASR training set, used for the few-shot experiments:
- `s2s.jsonl`: S -> S direction
- `s2t.jsonl`: S -> T direction
- `t2t.jsonl`: T -> T direction
- `t2s.jsonl`: T -> S direction

### Auto-Eval Speech And Text Classifiers

The sentiment of the generated sequence is estimated in an auto-eval fashion with Speech and Text classifiers. We point to the [paper](https://arxiv.org/abs/2402.05755) for details on these classifiers.


## Prediction & Evaluation of Spirit LM on STSP (Speech/Text)

```export PYTHONPATH=.```

Set `spiritlm` to the model you want to evaluate: e.g. ```spiritlm=spirit-lm-base-7b``` or ```spiritlm=spirit-lm-expressive-7b```

#### Speech to Text
    torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --model $spiritlm --eval_manifest_path data/stsp_data/manifest/emov/emov.test.jsonl --eval --write_pred ./pred_s_t.jsonl --input_output speech_text
#### Text to Text
    torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --model $spiritlm --eval_manifest_path data/stsp_data/manifest/emov/emov.test.jsonl --eval --write_pred ./pred_t_t.jsonl --input_output text_text
#### Text to Speech
    torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --model $spiritlm --eval_manifest_path data/stsp_data/manifest/emov/emov.test.jsonl --eval --write_pred ./pred_t_s.jsonl --input_output text_speech
#### Speech to Speech
    torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --model $spiritlm --eval_manifest_path data/stsp_data/manifest/emov/emov.test.jsonl --eval --write_pred ./pred_s_s.jsonl --input_output speech_speech


### Post-hoc Evaluation

To evaluate the performance of a model different from SpiritLM, you can use the following evaluation script that takes as input a prediction.jsonl file.

```
python spiritlm/eval/eval_stsp.py --ref_file $REF_FILE --pred_file $pred_file
```

e.g.

```
python spiritlm/eval/eval_stsp.py \
--ref_file ./data/examples/demo.jsonl  \
--pred_file ./data/examples/pred.jsonl
> Accuracy: 100.00% for predictions ./data/examples/pred.jsonl
```
