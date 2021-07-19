# MTTOD

This is code for the paper "Improving End-to-End Task-Oriented Dialogue System with A Simple Auxiliary Task".

## Environment setting

Our python version is 3.6.9.

The dependencies can be installed through:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data Preprocessing

For the experiments, we use MultiWOZ2.0 and MultiWOZ2.1.
- (MultiWOZ2.0) annotated_user_da_with_span_full.json: A fully annotated version of the original MultiWOZ2.0 data released by developers of Convlab available [here](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz/annotation).
- (MultiWOZ2.1) data.json: The original MultiWOZ 2.1 data released by researchers in University of Cambrige available [here](https://github.com/budzianowski/multiwoz/tree/master/data).

We rewrap the preprocessing scripts implemented by [Zhang et al., 2020](https://arxiv.org/abs/1911.10484). Please refer to [here](https://github.com/thu-spmi/damd-multiwoz/blob/master/data/multi-woz/README.md) for the details.

```
python preprocess.py -version 2.1
```

If you want to preprocess MultiWOZ2.0 data, set '-version' to '2.0'.

## Training

Our implementation supports a single GPU. Please use smaller batch sizes if out-of-memory error raises.

### MTTOD without auxiliary task (for the ablation)
```
python main.py -run_type train -model_dir checkpoints/ablation
```

### MTTOD with auxiliary task
```
python main.py -run_type train -model_dir checkpoints/mttod -add_auxiliary_task
```

To use MultiWOZ2.0 data, please set '-version' to '2.0' (default version is set to 2.1).

The checkpoints will be saved whenever each training epoch is done (default training epochs is set to 10).

## Inference

```
python main.py -run_type predict -ckpt checkpoints/mttod/ckpt-epoch10 -output greedy.json -batch_size 16
```

If you want to run other models, please set '-ckpt' to your checkpoint directory that includes model file (pytorch_model.bin).

The result file will be saved in the checkpoint directory.

To reduce inference time, it is recommended to set large 'batch_size'.

## Evaluation

We rewrap the evaluation scripts implemented by [Zhang et al., 2020](https://arxiv.org/abs/1911.10484).

```
python evaluator.py -data checkpoints/mttod/ckpt-epoch10/greedy.json
```

If you want to evaluate other result files, please set '-data' to your result file path.

## Acknowledgement

This code is adapted and modified upon the released code (https://github.com/thu-spmi/damd-multiwoz/) for "Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context".

For the pre-trained language model, we use huggingface's Transformer (https://huggingface.co/transformers/index.html#).

We are grateful for their excellent works.

## Bug report

Feel free to create an issue or send email to carep@etri.re.kr

