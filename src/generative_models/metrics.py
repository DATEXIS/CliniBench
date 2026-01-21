import logging

import torch
from sklearn.preprocessing import LabelEncoder
from torchmetrics.classification import Recall, Precision, Accuracy, F1Score
from torchmetrics.collections import MetricCollection
from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision, RetrievalMAP
from torchmetrics.wrappers import ClasswiseWrapper

torch.set_num_threads(1)

logger = logging.getLogger(__name__)


def contains_prefix(predicted_code, annotated):
    if predicted_code == "":
        return False
    if predicted_code[0].isalpha():
        if predicted_code[0:4] in annotated:
            return True
        else:
            return False
    else:
        if predicted_code[0:3] in annotated:
            return True
        else:
            return False


def short_code_from_code(code, icd_version=9):
    code = code.strip()
    if icd_version == 9:
        if code[0].isalpha():
            if code[0].lower() == "e":
                return code[:4]
            else:
                return code[:3]
        else:
            return code[:3]
    else:
        return code[:3]


def evaluate(results: dict):
    annotated_codes = results["unique_labels"]
    logging.getLogger().setLevel(logging.DEBUG)
    le = LabelEncoder()
    le.fit(list(annotated_codes))
    num_labels = len(annotated_codes)
    classification_metric_collection = MetricCollection(
        {
            "Multilabel-Recall-Macro": Recall(
                num_labels=num_labels, average="macro", task="multilabel"
            ).to("cpu"),
            "Multilabel-Precision-Macro": Precision(
                num_labels=num_labels, average="macro", task="multilabel"
            ).to("cpu"),
            "Multilabel-Accuracy-Macro": Accuracy(
                num_labels=num_labels, average="macro", task="multilabel"
            ).to("cpu"),
            "Multilabel-F1-Macro": F1Score(
                num_labels=num_labels, average="macro", task="multilabel"
            ).to("cpu"),
            "Multilabel-Recall-Micro": Recall(
                num_labels=num_labels, average="micro", task="multilabel"
            ).to("cpu"),
            "Multilabel-Precision-Micro": Precision(
                num_labels=num_labels, average="micro", task="multilabel"
            ).to("cpu"),
            "Multilabel-Accuracy-Micro": Accuracy(
                num_labels=num_labels, average="micro", task="multilabel"
            ).to("cpu"),
        }
    )

    classification_metric_collection_main_diagnosis = MetricCollection(
        {
            "Main Diagnosis Recall Macro": Recall(
                num_classes=num_labels, average="macro", task="multiclass"
            ).to("cpu"),
            "Main Diagnosis Precision Macro": Precision(
                num_classes=num_labels, average="macro", task="multiclass"
            ).to("cpu"),
            "Main Diagnosis Accuracy Macro": Accuracy(
                num_classes=num_labels, average="macro", task="multiclass"
            ).to("cpu"),
            "Main Diagnosis Recall Micro": Recall(
                num_classes=num_labels, average="micro", task="multiclass"
            ).to("cpu"),
            "Main Diagnosis Precision Micro": Precision(
                num_classes=num_labels, average="micro", task="multiclass"
            ).to("cpu"),
            "Main Diagnosis Accuracy Micro": Accuracy(
                num_classes=num_labels, average="micro", task="multiclass"
            ).to("cpu"),
        }
    )

    retrieval_metric_collection = MetricCollection(
        {
            "Recall@20": RetrievalRecall(top_k=20).to("cpu"),
            "Recall@10": RetrievalRecall(top_k=10).to("cpu"),
            "Recall@5": RetrievalRecall(top_k=5).to("cpu"),
            "Recall@3": RetrievalRecall(top_k=3).to("cpu"),
            "Recall@1": RetrievalRecall(top_k=1).to("cpu"),
            "Precision@20": RetrievalPrecision(top_k=20).to("cpu"),
            "Precision@10": RetrievalPrecision(top_k=10).to("cpu"),
            "Precision@5": RetrievalPrecision(top_k=5).to("cpu"),
            "Precision@3": RetrievalPrecision(top_k=3).to("cpu"),
            "Precision@1": RetrievalPrecision(top_k=1).to("cpu"),
            "MAP": RetrievalMAP().to("cpu"),
        }
    )
    all_labels = []
    all_main_labels = []
    all_predictions = []
    all_retr_predictions = []
    for i in range(len(results["predicted_codes"])):
        short_predicted_codes = [
            short_code_from_code(c, results["icd_version"])
            for c in results["predicted_codes"][i]
        ]

        # deduplicate:
        short_predicted_codes_cls = list(
            set(short_predicted_codes)
        )  # len(results["annotated_codes"][i])]
        short_predictions_idxs = torch.tensor(
            le.transform(short_predicted_codes_cls), dtype=torch.int32
        )
        short_predictions = torch.zeros((num_labels), dtype=torch.float32)
        short_predictions[short_predictions_idxs] = 1

        short_predicted_codes_retr = list(set(short_predicted_codes))
        short_predictions_idxs = torch.tensor(
            le.transform(short_predicted_codes_retr), dtype=torch.int32
        )
        short_predictions_retr = torch.zeros((num_labels), dtype=torch.float32)
        vals = torch.flip(
            torch.arange(1, len(short_predictions_idxs) + 1, dtype=torch.float32),
            dims=(0,),
        ) / float(len(short_predictions_idxs))
        short_predictions_retr[short_predictions_idxs] = vals

        label_idxs = torch.tensor(le.transform(results["annotated_codes"][i]))
        label_idx_main = torch.tensor(le.transform([results["annotated_codes"][i][0]]))

        labels = torch.zeros_like(short_predictions, dtype=torch.long)
        labels_main = torch.zeros_like(short_predictions, dtype=torch.long)

        labels[label_idxs] = 1
        # labels_main[label_idx_main] = 1
        all_labels.append(labels)
        all_main_labels.append(label_idx_main)
        all_predictions.append(short_predictions)
        all_retr_predictions.append(short_predictions_retr)

    indexes = torch.arange(len(all_predictions), dtype=torch.long)
    indexes = torch.repeat_interleave(indexes, repeats=num_labels, dim=0)

    all_predictions = torch.stack(all_predictions)
    all_retr_predictions = torch.stack(all_retr_predictions)

    all_labels = torch.stack(all_labels)
    all_main_labels = torch.stack(all_main_labels)

    classification_metric_collection(all_predictions.to("cpu"), all_labels.to("cpu"))

    retrieval_metric_collection(
        all_retr_predictions.to("cpu"),
        all_labels.to("cpu"),
        indexes=indexes.reshape_as(all_predictions).to("cpu"),
    )
    logger.info("Applied all classification and retrieval metrics")

    classwise_recall = ClasswiseWrapper(
        Recall(num_labels=num_labels, average=None, task="multilabel").to("cpu"),
        labels=list(annotated_codes),
        prefix="Recall-",
    )
    classwise_recall(all_predictions.to("cpu"), all_labels.to("cpu"))

    classwise_precision = ClasswiseWrapper(
        Precision(num_labels=num_labels, average=None, task="multilabel").to("cpu"),
        labels=list(annotated_codes),
        prefix="Precision-",
    )
    classwise_precision(all_predictions.to("cpu"), all_labels.to("cpu"))
    logger.info("Applied all classwise metrics")

    metrics = classification_metric_collection.compute()
    logger.info("Computed classification metrics")

    retrieval_metrics = retrieval_metric_collection.compute()
    logger.info("Computed retrieval metrics")

    logger.info("Computed main diagnosis classification metrics")

    classwise_recall = classwise_recall.compute()
    classwise_precision = classwise_precision.compute()
    logger.info("Computed classwise metrics")

    main_diagnosis_accuracy = torch.sum(
        all_predictions[
            torch.arange(0, len(all_main_labels)), all_main_labels.squeeze()
        ]
    ).float() / len(all_main_labels)

    for k, v in retrieval_metrics.items():
        metrics[k] = float(v)
    metrics["Main Diagnosis Accuracy"] = float(main_diagnosis_accuracy)

    for k, v in classwise_recall.items():
        metrics[k] = float(v)

    for k, v in classwise_precision.items():
        metrics[k] = float(v)

    for k, v in metrics.items():
        metrics[k] = float(v)

    return metrics
