"""
evaluate.py
-----------
RAG evaluation pipeline using DeepEval.

Measures three critical metrics:
  1. Answer Relevancy — is the answer actually relevant to the question?
  2. Faithfulness — does the answer stick to what the context says? (no hallucination)
  3. Contextual Relevancy — did the retriever pull the right chunks?

Usage:
  pip install deepeval
  deepeval test run evaluate.py

Or run directly:
  python evaluate.py
"""

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

# ── Test Cases ───────────────────────────────────────────────────────
# Replace these with real Q&A pairs from your documents.
# The more test cases, the more reliable your evaluation.

TEST_CASES = [
    {
        "input": "What is the refund policy?",
        "actual_output": "The refund policy allows returns within 30 days of purchase with a valid receipt.",
        "retrieval_context": [
            "Refund Policy: Customers may return items within 30 days of purchase. A valid receipt is required for all returns. Refunds are processed within 5-7 business days."
        ],
    },
    {
        "input": "What are the working hours?",
        "actual_output": "The office is open Monday through Friday, 9 AM to 5 PM.",
        "retrieval_context": [
            "Office Hours: Our standard working hours are Monday to Friday, 9:00 AM to 5:00 PM EST. The office is closed on weekends and federal holidays."
        ],
    },
    {
        "input": "What programming languages does the team use?",
        "actual_output": "I couldn't find that in the document.",
        "retrieval_context": [
            "Company Overview: Founded in 2015, the company specializes in data analytics and business intelligence solutions."
        ],
    },
]


# ── Metrics ──────────────────────────────────────────────────────────

relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
faithfulness_metric = FaithfulnessMetric(threshold=0.7)
contextual_metric = ContextualRelevancyMetric(threshold=0.7)


def test_answer_relevancy():
    """Test that answers are relevant to the questions asked."""
    for case in TEST_CASES:
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            retrieval_context=case["retrieval_context"],
        )
        assert_test(test_case, [relevancy_metric])


def test_faithfulness():
    """Test that answers don't hallucinate beyond the context."""
    for case in TEST_CASES:
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            retrieval_context=case["retrieval_context"],
        )
        assert_test(test_case, [faithfulness_metric])


def test_contextual_relevancy():
    """Test that retrieved context is relevant to the question."""
    for case in TEST_CASES:
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            retrieval_context=case["retrieval_context"],
        )
        assert_test(test_case, [contextual_metric])


if __name__ == "__main__":
    print("Running RAG evaluation...")
    print("=" * 60)

    total = len(TEST_CASES)
    passed = 0
    failed = 0

    for i, case in enumerate(TEST_CASES):
        print(f"\nTest {i+1}/{total}: {case['input']}")
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            retrieval_context=case["retrieval_context"],
        )
        try:
            assert_test(test_case, [relevancy_metric, faithfulness_metric])
            print("  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print(f"Pass rate: {passed/total*100:.0f}%")
