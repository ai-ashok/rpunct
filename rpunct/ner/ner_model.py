from __future__ import absolute_import, division, print_function

import collections
import logging
import os
import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
)
from transformers.convert_graph_to_onnx import convert

from .config.model_args import NERArgs
from .config.utils import sweep_config_to_sweep_values
from .ner_utils import (
    InputExample,
    LazyNERDataset,
    convert_examples_to_features,
    get_examples_from_df,
    load_hf_dataset,
    read_examples_from_file,
)


wandb_available = False

logger = logging.getLogger(__name__)

MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT = ["squeezebert", "deberta", "mpnet"]

MODELS_WITH_EXTRA_SEP_TOKEN = [
    "roberta",
    "camembert",
    "xlmroberta",
    "longformer",
    "mpnet",
]


class NERModel:
    def __init__(
        self,
        model_type,
        model_name,
        labels=None,
        weight=None,
        args=None,
        use_cuda=torch.cuda.is_available(),
        cuda_device=-1,
        **kwargs,
    ):
        """
        Initializes a NERModel

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_model.bin).
            labels (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
            weight (optional): A `torch.Tensor`, `numpy.ndarray` or list.  The weight to be applied to each class when computing the loss of the model.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): The execution provider to use for ONNX export.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, NERArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if not use_cuda:
            self.args.fp16 = False

        if labels and self.args.labels_list:
            assert labels == self.args.labels_list
            self.args.labels_list = labels
        elif labels:
            self.args.labels_list = labels
        elif self.args.labels_list:
            pass
        else:
            self.args.labels_list = [
                "O",
                "B-MISC",
                "I-MISC",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
            ]
        self.num_labels = len(self.args.labels_list)
        self.id2label = {i: label for i, label in enumerate(self.args.labels_list)}
        self.label2id = {label: i for i, label in enumerate(self.args.labels_list)}

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if self.num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=self.num_labels, **self.args.config
            )
            self.num_labels = self.num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        self.config.id2label = self.id2label
        self.config.label2id = self.label2id

        if model_type in MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT and weight is not None:
            raise ValueError(
                "{} does not currently support class weights".format(model_type)
            )
        else:
            self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if not self.args.quantized_model:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, **kwargs
            )
        else:
            quantized_weights = torch.load(
                os.path.join(model_name, "pytorch_model.bin")
            )
            self.model = model_class.from_pretrained(
                None, config=self.config, state_dict=quantized_weights
            )

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

        self.results = {}

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError(
                    "fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16."
                )

        if model_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name,
                do_lower_case=self.args.do_lower_case,
                normalization=True,
                **kwargs,
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

    def predict(self, to_predict, split_on_space=True):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
            split_on_space: If True, each sequence will be split by spaces for assigning labels.
                            If False, to_predict must be a a list of lists, with the inner list being a
                            list of strings consisting of the split sequences. The outer list is the list of sequences to
                            predict on.

        Returns:
            preds: A Python list of lists with dicts containing each word mapped to its NER tag.
            model_outputs: A Python list of lists with dicts containing each word mapped to its list with raw model output.
        """  # noqa: ignore flake8"

        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        preds = None

        if split_on_space:
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                predict_examples = [
                    InputExample(
                        i,
                        sentence.split(),
                        [self.args.labels_list[0] for word in sentence.split()],
                        x0,
                        y0,
                        x1,
                        y1,
                    )
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(
                        i,
                        sentence.split(),
                        [self.args.labels_list[0] for word in sentence.split()],
                    )
                    for i, sentence in enumerate(to_predict)
                ]
        else:
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                predict_examples = [
                    InputExample(
                        i,
                        sentence,
                        [self.args.labels_list[0] for word in sentence],
                        x0,
                        y0,
                        x1,
                        y1,
                    )
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(
                        i, sentence, [self.args.labels_list[0] for word in sentence]
                    )
                    for i, sentence in enumerate(to_predict)
                ]

        eval_dataset = self.load_and_cache_examples(None, to_predict=predict_examples)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        self._move_model_to_device()

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        for batch in tqdm(
            eval_dataloader, disable=args.silent, desc="Running Prediction"
        ):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = self._calculate_loss(model, inputs)
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = self._calculate_loss(model, inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
                out_input_ids = np.append(
                    out_input_ids,
                    inputs["input_ids"].detach().cpu().numpy(),
                    axis=0,
                )
                out_attention_mask = np.append(
                    out_attention_mask,
                    inputs["attention_mask"].detach().cpu().numpy(),
                    axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps

        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        if split_on_space:
            preds = [
                [
                    {word: preds_list[i][j]}
                    for j, word in enumerate(sentence.split()[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            preds = [
                [
                    {word: preds_list[i][j]}
                    for j, word in enumerate(sentence[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]

        word_tokens = []
        for n, sentence in enumerate(to_predict):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[n],
                out_label_ids[n],
                out_attention_mask[n],
                token_logits[n],
            )
            word_tokens.append(w_log)

        if split_on_space:
            model_outputs = [
                [
                    {word: word_tokens[i][j]}
                    for j, word in enumerate(sentence.split()[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            model_outputs = [
                [
                    {word: word_tokens[i][j]}
                    for j, word in enumerate(sentence[: len(preds_list[i])])
                ]
                for i, sentence in enumerate(to_predict)
            ]

        return preds, model_outputs

    def _convert_tokens_to_word_logits(
        self, input_ids, label_ids, attention_mask, logits
    ):

        ignore_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
        ]

        # Remove unuseful positions
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        # Map to word logits
        word_logits = []
        tmp = []
        for n, lab in enumerate(masked_labels):
            if lab != self.pad_token_label_id:
                if n != 0:
                    word_logits.append(tmp)
                tmp = [list(masked_logits[n])]
            else:
                tmp.append(list(masked_logits[n]))
        word_logits.append(tmp)

        return word_logits

    def load_and_cache_examples(
        self, data, evaluate=False, no_cache=False, to_predict=None
    ):
        """
        Reads data_file and generates a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.

        Args:
            data: Path to a .txt file containing training or evaluation data OR a pandas DataFrame containing 3 columns - sentence_id, words, labels.
                    If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            evaluate (optional): Indicates whether the examples are for evaluation or for training.
            no_cache (optional): Force feature conversion and prevent caching. I.e. Ignore cached features even if present.

        """  # noqa: ignore flake8"

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        mode = "dev" if evaluate else "train"
        if self.args.use_hf_datasets and data is not None:
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            dataset = load_hf_dataset(
                data,
                self.args.labels_list,
                self.args.max_seq_length,
                self.tokenizer,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token_id,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token_id,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=self.pad_token_label_id,
                silent=args.silent,
                args=self.args,
            )
        else:
            if not to_predict and isinstance(data, str) and self.args.lazy_loading:
                dataset = LazyNERDataset(data, tokenizer, self.args)
            else:
                if to_predict:
                    examples = to_predict
                    no_cache = True
                else:
                    if isinstance(data, str):
                        examples = read_examples_from_file(
                            data,
                            mode,
                            bbox=True
                            if self.args.model_type in ["layoutlm", "layoutlmv2"]
                            else False,
                        )
                    else:
                        if self.args.lazy_loading:
                            raise ValueError(
                                "Input must be given as a path to a file when using lazy loading"
                            )
                        examples = get_examples_from_df(
                            data,
                            bbox=True
                            if self.args.model_type in ["layoutlm", "layoutlmv2"]
                            else False,
                        )

                cached_features_file = os.path.join(
                    args.cache_dir,
                    "cached_{}_{}_{}_{}_{}".format(
                        mode,
                        args.model_type,
                        args.max_seq_length,
                        self.num_labels,
                        len(examples),
                    ),
                )
                if not no_cache:
                    os.makedirs(self.args.cache_dir, exist_ok=True)

                if os.path.exists(cached_features_file) and (
                    (not args.reprocess_input_data and not no_cache)
                    or (
                        mode == "dev" and args.use_cached_eval_features and not no_cache
                    )
                ):
                    features = torch.load(cached_features_file)
                    logger.info(
                        f" Features loaded from cache at {cached_features_file}"
                    )
                else:
                    logger.info(" Converting to features started.")
                    features = convert_examples_to_features(
                        examples,
                        self.args.labels_list,
                        self.args.max_seq_length,
                        self.tokenizer,
                        # XLNet has a CLS token at the end
                        cls_token_at_end=bool(args.model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        # RoBERTa uses an extra separator b/w pairs of sentences,
                        # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                        # PAD on the left for XLNet
                        pad_on_left=bool(args.model_type in ["xlnet"]),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                        pad_token_label_id=self.pad_token_label_id,
                        process_count=process_count,
                        silent=args.silent,
                        use_multiprocessing=args.use_multiprocessing,
                        chunksize=args.multiprocessing_chunksize,
                        mode=mode,
                        use_multiprocessing_for_evaluation=args.use_multiprocessing_for_evaluation,
                    )

                    if not no_cache:
                        torch.save(features, cached_features_file)

                all_input_ids = torch.tensor(
                    [f.input_ids for f in features], dtype=torch.long
                )
                all_input_mask = torch.tensor(
                    [f.input_mask for f in features], dtype=torch.long
                )
                all_segment_ids = torch.tensor(
                    [f.segment_ids for f in features], dtype=torch.long
                )
                all_label_ids = torch.tensor(
                    [f.label_ids for f in features], dtype=torch.long
                )

                if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                    all_bboxes = torch.tensor(
                        [f.bboxes for f in features], dtype=torch.long
                    )

                if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                    dataset = TensorDataset(
                        all_input_ids,
                        all_input_mask,
                        all_segment_ids,
                        all_label_ids,
                        all_bboxes,
                    )
                else:
                    dataset = TensorDataset(
                        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
                    )

        return dataset

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(
                    output_dir
                )
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="ner",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self._save_model_args(output_dir)

    def _calculate_loss(self, model, inputs):
        outputs = model(**inputs)
        # model outputs are always tuple in pytorch-transformers (see doc)
        loss = outputs[0]
        return (loss, *outputs[1:])

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        if self.args.use_hf_datasets and isinstance(batch, dict):
            return {key: value.to(self.device) for key, value in batch.items()}
        else:
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if self.args.model_type in [
                "bert",
                "xlnet",
                "albert",
                "layoutlm",
                "layoutlmv2",
            ]:
                inputs["token_type_ids"] = batch[2]

            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                inputs["bbox"] = batch[4]

            return inputs

    def _create_training_progress_scores(self, **kwargs):
        return collections.defaultdict(list)
        """extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }

        return training_progress_scores"""

    def save_model(
        self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            self._save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = NERArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
