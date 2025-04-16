# utils/evaluation.py
from langchain.smith import RunEvalConfig
from langchain.smith.evaluation import StringEvaluator
from langchain.smith.evaluation.string_evaluators import (
    LabeledCriteriaEvaluator,
)

def setup_evaluation():
    # Configure evaluation criteria
    eval_config = RunEvalConfig(
        evaluators=[
            StringEvaluator(
                "hallucination",
                {
                    "criteria": "Does the response contain information not supported by the provided sources?",
                    "labels": ["yes", "no"],
                },
            ),
            StringEvaluator(
                "faithfulness",
                {
                    "criteria": "Does the response accurately represent the information from the sources?",
                    "labels": ["yes", "no"],
                },
            ),
            StringEvaluator(
                "relevance",
                {
                    "criteria": "Is the response relevant to the original query?",
                    "labels": ["yes", "no"],
                },
            ),
            StringEvaluator(
                "completeness",
                {
                    "criteria": "Does the response address all aspects of the query?",
                    "labels": ["yes", "no"],
                },
            ),
        ]
    )
    return eval_config
