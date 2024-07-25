from collections import Counter

import dspy
from dsp.utils import normalize_text, print_message
from dspy import ColBERTv2, OpenAI, settings
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch


# Define F1 metric calculation
def f1_score_01(prediction, ground_truth):
    prediction_tokens = [normalize_text(elem) for elem in prediction.split("|")]
    ground_truth_tokens = [normalize_text(elem) for elem in ground_truth.split("|")]

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        print_message(
            "\n#> F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\n"
        )

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1


# Define F1 function to handle a list of ground truth answers
def F1(prediction, answers_list):
    assert isinstance(answers_list, list)
    return max(f1_score_01(prediction, ans) for ans in answers_list)


# Define answer matching using F1 score
def answer_match(prediction, answers, frac=1.0):
    return F1(prediction, answers) >= frac


# Define the main validation function
def answer_f1_match_01(example, pred, trace=None, frac=0.95):
    assert isinstance(example.answer, (str, list))
    if isinstance(example.answer, str):
        return answer_match(pred.answer, [example.answer], frac=frac)
    else:  # example.answer is a list
        return answer_match(pred.answer, example.answer, frac=frac)


# Define NewsCategorization signature
class NewsCategorization(dspy.Signature):
    """
    Categorize News body as 'fake' or 'real'.

    This task aims to understand the nature of the news content and classify it accordingly.
    The classification should consider the language and any indicative markers that
    suggest whether the news was generated automatically or crafted by a human.
    """

    news_body = dspy.InputField(desc="The body of the news to be categorized")
    answer = dspy.OutputField(desc="Should be 'fake' or 'real'")


# Define CoTCombined module
class CoTCombined(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(NewsCategorization)

    def forward(self, news_body):
        pred_list = []
        for news in news_body.split("|"):
            pred_one = self.prog(news_body=news)
            pred_list.append(pred_one.answer)
        return dspy.Prediction(answer="|".join(pred_list))


# Define CustomExample class
class CustomExample:
    def __init__(self, news_body, answer):
        self.news_body = news_body
        self.answer = answer

    def with_inputs(self, input_key):
        return self

    def inputs(self):
        return {"news_body": self.news_body}

    def items(self):
        return {"news_body": self.news_body, "answer": self.answer}.items()

    def copy(self):
        return CustomExample(self.news_body, self.answer)

    def get(self, key, default=None):
        return {"news_body": self.news_body, "answer": self.answer}.get(key, default)

    def __iter__(self):
        return iter({"news_body": self.news_body, "answer": self.answer})

    def __contains__(self, key):
        return key in {"news_body": self.news_body, "answer": self.answer}

    def __getitem__(self, key):
        return {"news_body": self.news_body, "answer": self.answer}[key]


# Define custom train and dev sets
custom_trainset = [
    CustomExample("Fake news body 1", "fake"),
    CustomExample("Real news body 2", "real"),
    CustomExample("Fake news body 3", "fake"),
]

custom_devset = [
    CustomExample("Real news body 4", "real"),
    CustomExample("Fake news body 5", "fake"),
]

# Configure DSPy settings
api_key = "your-valid-api-key"
llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=2000)
colbertv2_wiki17_abstracts = ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts)

# Initialize the teleprompter with the validation function
teleprompter = BootstrapFewShotWithRandomSearch(metric=answer_f1_match_01)

# Compile the RAG program
compiled_rag = teleprompter.compile(CoTCombined(), trainset=custom_trainset)

# Test the compiled program with a news body
my_news_body = "This is a fake news body.|This is a real news body."
pred = compiled_rag(my_news_body)
print(f"News Body: {my_news_body}")
print(f"Predicted Answer: {pred.answer}")

# Evaluate the compiled program on the custom devset
evaluate_on_custom_devset = Evaluate(
    devset=custom_devset, num_threads=1, display_progress=False, display_table=5
)
evaluation_results = evaluate_on_custom_devset(compiled_rag, metric=answer_f1_match_01)

print("Evaluation Results:")
print(evaluation_results)
