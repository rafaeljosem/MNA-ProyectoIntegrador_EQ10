from deepeval import assert_test, evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()


def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost."
    )
    assert_test(test_case, [answer_relevancy_metric])


def test_contextual_recall():

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."

    # Replace this with the expected output from your RAG generator
    expected_output = "You are eligible for a 30 day full refund at no extra cost."

    # Replace this with the actual retrieved context from your RAG pipeline
    retrieval_context = [
        "All customers are eligible for a 30 day full refund at no extra cost."]

    metric = ContextualRecallMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    metric.measure(test_case)
    print(metric.score)
    print(metric.reason)

    # or evaluate test cases in bulk
    evaluate([test_case], [metric])
