"""
Converts a normal dataset to a timeml format dataset with event classification.

Inspired by "Exploring Pre-trained Language Models for Event Extraction and Generation" 
(Yang et. al., ACL 2019) (https://www.aclweb.org/anthology/P19-1522.pdf)

Settings recommended by the BERT's authors:
    * Batch size: 16, 32
    * Learning rate (Adam): 5e-5, 3e-5, 2e-5
    * Number of epochs: 2, 3, 4

Todo:
* finish annotate_one_instance [Done]
* hierarchical classification [Done]
* delete huggingface model related code [Done]
* change accuracy matrix to ignore padding [Done]
* different loss weights for the 1st and 2nd levels [Done]
* train the 1st level for 2 epochs, and then train the 2nd one by freezing the remaining part [Done]
    - To ensure the lower bound of accuracy [Done]
* solve overfitting: dropout? 1 epoch only? [Done]
* add F1 score [Done]
* compare with ternary classification [Done]
* update the parser version [Done]
"""
import datetime
import json
import os
import random
import time

from typing import List, Tuple
import numpy as np
import torch
from lxml import etree
from spacy.lang.en import English
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    random_split,
)
from tqdm import tqdm
from transformers import (
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from templi.dataset_converters.timeml_parser import parser
from templi.models.temli_multiway_match import (
    BertHierarchicalEventAnnotator,
    BertTernaryEventAnnotator,
)


def _get_bert_token_indexes_no_spacing(tokens):
    i = 0
    indexes = []
    for token in tokens[1:-1]:
        tok_len = 1 if token == "[UNK]" else len(token)
        if token.startswith("##"):
            tok_len -= 2
        indexes.append((i, i + tok_len))
        i += tok_len
    return indexes


def make_dataloaders(tokenizer, sentences_anno, one_batch_for_test=False):
    def get_labels(tokens, annotations):
        """For bert-tokenized tokens only"""
        indexes = _get_bert_token_indexes_no_spacing(tokens)
        annotations_sorted = sorted(annotations, key=lambda x: x[0])
        labels = torch.zeros(max_seq_len, dtype=torch.long)
        ends = torch.zeros(max_seq_len, dtype=torch.long)
        s, e = -1, -1
        for i, (si, ei) in enumerate(indexes):
            if si >= e and len(annotations_sorted) > 0:
                s, e = annotations_sorted.pop(0)
            labels[i + 1] = 1 if si >= s and ei <= e else 0
            ends[i + 1] = 1 if ei == e else 0
        # print(' '.join([t for t, l in zip(tokens, labels.tolist()) if l == 1]))  # debug
        return labels, ends

    def make_all_input_tensors():
        ctr_truncated = 0
        n_samples = (
            len(list(sentences_anno.keys())) if not one_batch_for_test else batch_size
        )
        all_input_ids = torch.zeros(n_samples, max_seq_len, dtype=torch.long)
        all_attention_mask = torch.zeros(n_samples, max_seq_len, dtype=torch.long)
        all_labels = torch.zeros(n_samples, max_seq_len, dtype=torch.long)
        all_ends = torch.zeros(n_samples, max_seq_len, dtype=torch.long)
        for i, (sentence, annotations) in tqdm(
            enumerate(sentences_anno.items()), desc="produce labels"
        ):
            if one_batch_for_test and i == batch_size:
                break

            # DEBUG
            # print(sentence)
            # for s,e in annotations.keys():
            #     print('\t', sentence.replace(' ','')[s:e])

            tokenizer_output = tokenizer(sentence, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(
                tokenizer_output["input_ids"].squeeze(0)
            )  # single instance input, not batch
            if len(tokens) > max_seq_len:  # hack: truncate long input
                ctr_truncated += 1
                tokens = tokens[: max_seq_len - 1] + tokens[-1:]
                all_input_ids[i][: max_seq_len - 1] = tokenizer_output["input_ids"][0][
                    : max_seq_len - 1
                ]
                all_input_ids[i][-1:] = tokenizer_output["input_ids"][0][-1:]
                all_attention_mask[i][:] = tokenizer_output["attention_mask"][0][
                    :max_seq_len
                ]
            else:
                all_input_ids[i][: len(tokens)] = tokenizer_output["input_ids"][0]
                all_attention_mask[i][: len(tokens)] = tokenizer_output[
                    "attention_mask"
                ][0]
            all_labels[i], all_ends[i] = get_labels(tokens, annotations)
        print(
            "{} out of {} input sentences truncated ({:.2f}%).".format(
                ctr_truncated,
                len(sentences_anno.items()),
                ctr_truncated / len(sentences_anno.items()) * 100,
            )
        )
        return all_input_ids, all_attention_mask, all_labels, all_ends

    all_input_ids, all_attention_mask, all_labels, all_ends = make_all_input_tensors()

    # Dataloader
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_ends)

    if one_batch_for_test:
        return DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size,
        )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("{:>5,} training samples".format(train_size))
    print("{:>5,} validation samples".format(val_size))

    # Train / valid split
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,  # Evaluate with this batch size.
    )

    return train_dataloader, validation_dataloader


def run(train_dataloader, validation_dataloader, model):
    """
    Adapted from: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    """

    # Function to calculate the accuracy of our predictions vs labels
    def flat_f1_score(preds, labels, input_mask):
        mask = input_mask.flatten()
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        tp = np.sum((pred_flat == 1) * (labels_flat == 1) * (mask == 1))
        fp = np.sum((pred_flat == 1) * (labels_flat == 0) * (mask == 1))
        fn = np.sum((pred_flat == 0) * (labels_flat == 1) * (mask == 1))
        precision = tp / (tp + fp) if tp > 0 else 0
        recall = tp / (tp + fn) if tp > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision * recall > 0
            else 0
        )
        return f1_score

    def flat_accuracy(preds, labels, input_mask):
        mask = input_mask.flatten()
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum((pred_flat == labels_flat) * (mask == 1)) / np.sum(mask == 1)

    def format_time(elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        """
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        eps=adamw_eps,
    )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss, avg_train_loss, training_time = 0, 0, 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Avg loss: {:.3f}".format(
                        step, len(train_dataloader), elapsed, total_train_loss / step
                    )
                )

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels_1 = batch[2].to(device)
            b_labels_2 = batch[3].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            if epoch_i in epochs_bert_1st:
                train_parts = ["bert", "1st-level"]
            elif epoch_i in epochs_2nd:
                train_parts = ["2nd-level"]
            else:
                train_parts = ["bert", "1st-level", "2nd-level"]

            # Freeze some parts of the model as specified
            model.bert.requires_grad_("bert" in train_parts)
            model.classifier_1.requires_grad_("1st-level" in train_parts)
            model.classifier_2.requires_grad_("2nd-level" in train_parts)

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits_1, logits_2 = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels_1=b_labels_1,
                labels_2=b_labels_2,
                return_dict=False,
                second_level_loss="2nd-level" in train_parts,
            )

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy_1, total_eval_f1_score_1 = 0, 0
        total_eval_accuracy_2, total_eval_f1_score_2 = 0, 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels_1 = batch[2].to(device)
            b_labels_2 = batch[3].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, logits_1, logits_2 = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels_1=b_labels_1,
                    labels_2=b_labels_2,
                    return_dict=False,
                    second_level_loss="2nd-level" in train_parts,
                )

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits_1 = logits_1.detach().cpu().numpy()
            logits_2 = logits_2.detach().cpu().numpy()
            label_ids_1 = b_labels_1.to("cpu").numpy()
            label_ids_2 = b_labels_2.to("cpu").numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy_1 += flat_accuracy(
                logits_1, label_ids_1, b_input_mask.to("cpu").numpy()
            )
            total_eval_accuracy_2 += flat_accuracy(
                logits_2, label_ids_2, b_labels_1.to("cpu").numpy()
            )
            total_eval_f1_score_1 += flat_f1_score(
                logits_1, label_ids_1, b_input_mask.to("cpu").numpy()
            )
            total_eval_f1_score_2 += flat_f1_score(
                logits_2, label_ids_2, b_labels_1.to("cpu").numpy()
            )

        # Report the final accuracy for this validation run.
        avg_val_accuracy_1 = total_eval_accuracy_1 / len(validation_dataloader)
        avg_val_f1_score_1 = total_eval_f1_score_1 / len(validation_dataloader)
        print(
            "  Accuracy_1: {0:.2f}    F1_score_1: {1:.2f}".format(
                avg_val_accuracy_1, avg_val_f1_score_1
            )
        )

        avg_val_accuracy_2 = total_eval_accuracy_2 / len(validation_dataloader)
        avg_val_f1_score_2 = total_eval_f1_score_2 / len(validation_dataloader)
        print(
            "  Accuracy_2: {0:.2f}    F1_score_2: {1:.2f}".format(
                avg_val_accuracy_2, avg_val_f1_score_2
            )
        )

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur. 1": avg_val_accuracy_1,
                "Valid. F1 Scr 1": avg_val_f1_score_1,
                "Valid. Accur. 2": avg_val_accuracy_2,
                "Valid. F1 Scr 2": avg_val_f1_score_2,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

    print("")
    print("Training complete!")

    print(
        "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
    )

    return training_stats


def load_sentence_anno(data_dir, data_paths, small_subset_size=-1):
    sentences_anno = dict()
    for data_path in data_paths:
        file_dir = os.path.join(data_dir, data_path)
        file_list = os.listdir(file_dir)
        print("parse {}...".format(file_dir))
        for i, file in enumerate(tqdm(file_list, desc="parse xmls")):
            if i == small_subset_size:
                break
            with open(os.path.join(file_dir, file)) as f:
                news = f.read().replace("\n", "")
            new_annos = parser(news, return_only_sentences_anno=True)
            sentences_anno = {**sentences_anno, **new_annos}
    return sentences_anno


def save_model(model, model_out_dir, training_stats=None):
    if not os.path.isdir(model_out_dir):
        os.mkdir(model_out_dir)
    torch.save(model, os.path.join(model_out_dir, "model.pt"))

    if training_stats:
        with open(os.path.join(model_out_dir, "training_stats.json"), "w") as fout:
            json.dump(training_stats, fout, indent=4)


def annotate_one_instance(model, tokenizer, sentence: str) -> List[Tuple[int, int]]:
    """
    This function extracts the events from a sentence.
    """

    def reduce_annotations(annotations, ends):
        def try_append(s, e):
            if s != -1 and e != -1:
                new_anno.append((s, e))

        if len(annotations) <= 1:
            return annotations
        new_anno = []
        ps, pe = annotations[0]
        for s, e in annotations[1:]:
            if len(ends) > 0 and e == ends[0][1]:
                ends.pop(0)
                if s == pe:
                    try_append(ps, e)
                else:
                    try_append(ps, pe)
                    try_append(s, e)
                ps, pe = -1, -1
            elif s == pe:
                pe = e
            else:
                try_append(ps, pe)
                ps, pe = s, e
        try_append(ps, pe)
        return new_anno

    tokenizer_output = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(tokenizer_output["input_ids"].squeeze(0))
    indexes = _get_bert_token_indexes_no_spacing(tokens)

    model = model.to(device)
    logits_1, logits_2 = model(
        tokenizer_output["input_ids"].to(device),
        token_type_ids=None,
        attention_mask=tokenizer_output["attention_mask"].to(device),
        return_dict=False,
    )
    logits_1 = logits_1.squeeze(0).detach().cpu()
    logits_2 = logits_2.squeeze(0).detach().cpu()
    labels_1 = np.argmax(logits_1, axis=1)
    labels_2 = np.argmax(logits_2, axis=1)

    annotations = [
        idx for idx, label in zip(indexes, labels_1[1 : 1 + len(indexes)]) if label == 1
    ]
    ends = [
        idx
        for idx, label in zip(indexes, labels_2[1 : 1 + len(indexes)])
        if idx in annotations and label == 1
    ]
    annotations = reduce_annotations(annotations, ends)

    # DEBUG
    # raw_annotated_tokens = [t for t, l in zip(tokens, labels_1) if l == 1]
    # raw_annotated_ends = [t for t, l in zip(tokens, labels_2) if l == 1]
    # print(sentence)
    # print('\t', raw_annotated_tokens)
    # print('\t', raw_annotated_ends)

    return annotations


if __name__ == "__main__":
    max_seq_len = 180  # anything longer are truncated  (180 => 0.07% truncated)
    batch_size = 8  # 16 => CUDA out of memory
    device = "cuda"
    learning_rate = 5e-5
    adamw_eps = 1e-8  # for 10 docs, acc/val loss: 0.83/0.22 for 1e-7; [0.95/0.21 for 1e-8]; 0.82/0.17 for 1e-9; 0.79/0.12 for 1e-10
    # epochs = 2
    # epochs_bert_1st = []
    # epochs_2nd = []
    # eps_tag = "".join(
    #     [
    #         "1" if i in epochs_bert_1st else "2" if i in epochs_2nd else "0"
    #         for i in range(epochs)
    #     ]
    # )
    # config_tag = "-bert-base-uncased-hier(torch)-eps_{}".format(eps_tag)
    config_tag = '-bert-base-uncased-hier(torch)-eps_00'

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    data_dir = "/mnt/AAI_Project/_Dataset/timebank_1_2/data"
    data_paths = ["timeml", "extra"]
    model_out_dir = (
        "/mnt/AAI_Project/_Archive/try/temli_1.0/training/event-annotator" + config_tag
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load finetuning data
    # sentences_anno = load_sentence_anno(data_dir, data_paths, small_subset_size=-1)
    # train_dataloader, validation_dataloader = make_dataloaders(
    #     tokenizer, sentences_anno
    # )
    # Load testing data
    sentences_anno = load_sentence_anno(data_dir, data_paths, small_subset_size=1)
    test_dataloader = make_dataloaders(
        tokenizer, sentences_anno, one_batch_for_test=True
    )

    print("\n" + "#" * 20 + " " + config_tag + " " + "#" * 20)

    # Load model
    # model = BertHierarchicalEventAnnotator.from_pretrained("bert-base-uncased")
    model = torch.load(os.path.join(model_out_dir, "model.pt"))

    # Finetune
    # training_stats = run(train_dataloader, validation_dataloader, model)
    # save_model(model, model_out_dir, training_stats)

    # Annotate
    sentence = "I didn't think this annotator would work on Jan 14, but it does today."
    annotations = annotate_one_instance(
        model, tokenizer, sentence
    )

    # Test annotations
    sentence_no_spacing = sentence.replace(" ", "")
    annotated_events = [sentence_no_spacing[s:e] for s, e in annotations]
    print(annotated_events)