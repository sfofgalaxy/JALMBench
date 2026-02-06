# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def pred_to_label(
    sentiment_prediction_scores: List[List[Dict[str, Any]]],
) -> Tuple[str, float]:
    if isinstance(sentiment_prediction_scores[0], list):
        sentiment_prediction_scores = sentiment_prediction_scores[0]
    item_with_max_score = max(
        sentiment_prediction_scores, key=lambda _dict: _dict["score"]
    )
    score = item_with_max_score["score"]
    return score, item_with_max_score["label"].lower()


def get_text_sentiment_prediction(text: str, sentiment_classifier) -> Tuple[str, float]:
    return pred_to_label(sentiment_classifier(text))


def load_sentiment_classifier(model_dir: str):
    classifier = pipeline(
        task="text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(model_dir),
        tokenizer=AutoTokenizer.from_pretrained(
            "j-hartmann/sentiment-roberta-large-english-3-classes"
        ),
        top_k=None,
    )
    return classifier
