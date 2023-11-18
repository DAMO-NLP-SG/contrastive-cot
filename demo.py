import json
import random
import re
from pathlib import Path
from typing import List, TextIO

import openai
from datasets import load_dataset
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm


def remove_substrings_with_double_angle_brackets(input_string):
    # Define the pattern to match substrings within double angled brackets
    pattern = r"<<[^>]+>>"
    # Use the sub() function from the re module to replace matching substrings with an empty string
    result = re.sub(pattern, "", input_string)
    return result


class ReasonSample(BaseModel):
    question: str
    explanation: str = ""
    answer: str = ""
    wrong_explanation: str = ""
    wrong_answer: str = ""
    pred: str = ""


class ReasonData(BaseModel):
    samples: List[ReasonSample]

    @classmethod
    def load_gsm8k_test(cls, path: str = "gsm8k", subset: str = "main", split="test"):
        samples = []
        for raw in load_dataset(path, subset, split=split):
            explanation, answer = raw["answer"].split("####")
            explanation = remove_substrings_with_double_angle_brackets(explanation)
            samples.append(
                ReasonSample(
                    question=raw["question"].strip(),
                    explanation=explanation.strip(),
                    answer=answer.strip(),
                )
            )

        return cls(samples=samples)

    @classmethod
    def load_gsm8k_incoherent_objects(cls, split: str = "test", sample: bool = False):
        if split == "train" and sample:
            samples = [
                ReasonSample(
                    question="There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                    explanation="There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted.",
                    answer="6",
                    wrong_explanation="There are 21 - 15 = 6 trees originally. Then there were 15 trees after the Grove workers planted some more. So there must have been 21 trees that were planted.",
                    wrong_answer="21",
                ),
                ReasonSample(
                    question="If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot??",
                    explanation="There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot.",
                    answer="5",
                    wrong_explanation="There are originally 3 + 2 = 5 cars. Then 3 more cars arrive. Now 2 cars are in the parking lot",
                    wrong_answer="2",
                ),
                ReasonSample(
                    question="Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                    explanation="Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total.",
                    answer="39",
                    wrong_explanation="Originally, Leah had 32 + 42 = 74 chocolates and her sister had 32. So in total they had 74 - 35 = 39. After eating 35, they had 42 pieces left in total",
                    wrong_answer="42",
                ),
                ReasonSample(
                    question="Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                    explanation="Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.",
                    answer="8",
                    wrong_explanation="Jason had 20 - 12 = 8 lollipops originally. Then he had 20 after giving some to Denny. So he gave Denny 12 lollipops",
                    wrong_answer="12",
                ),
            ]
            return cls(samples=samples)
        else:
            return cls.load_gsm8k_test()

    @classmethod
    def load_from_name(cls, name: str, **kwargs):
        if name == "gsm8k":
            return cls.load_gsm8k_incoherent_objects(**kwargs)
        else:
            raise KeyError(name)


class Prompter(BaseModel):
    def run(self, data_train: ReasonData, sample_test: ReasonSample) -> str:
        prompt = ""
        for sample in data_train.samples:
            prompt += f"Question: {sample.question}\n"
            prompt += f"Answer: {sample.answer}\n\n"

        prompt += f"Question: {sample_test.question}\n"
        prompt += "Answer: "
        return prompt

    @staticmethod
    def get_answer(text: str) -> str:
        parts = text.split("Answer: ")
        if len(parts) >= 2:
            return parts[1]
        else:
            return text


class ChainThoughtPrompter(Prompter):
    def run(self, data_train: ReasonData, sample_test: ReasonSample) -> str:
        prompt = ""
        for sample in data_train.samples:
            prompt += f"Question: {sample.question}\n"
            prompt += f"Explanation: {sample.explanation}\n"
            prompt += f"Answer: {sample.answer}\n\n"

        prompt += f"Question: {sample_test.question}\n"
        prompt += "Explanation: "
        return prompt

    def get_explanation(self, text: str) -> str:
        assert self is not None
        return text.split("\nAnswer: ")[0]


class ContrastiveChainThoughtPrompter(Prompter):
    def run(self, data_train: ReasonData, sample_test: ReasonSample) -> str:
        prompt = ""
        for sample in data_train.samples:
            prompt += f"Question: {sample.question}\n"
            prompt += f"Explanation: {sample.explanation}\n"
            prompt += f"Answer: {sample.answer}\n"
            prompt += f"Wrong explanation: {sample.wrong_explanation}\n"
            prompt += f"Wrong Answer: {sample.wrong_answer}\n\n"

        prompt += f"Question: {sample_test.question}\n"
        prompt += "Explanation: "
        return prompt


def select_prompter(name: str):
    if name == "standard":
        return Prompter()
    if name == "cot":
        return ChainThoughtPrompter()
    if name == "contrast_cot":
        return ContrastiveChainThoughtPrompter()
    else:
        raise KeyError(name)


class OpenAIModel(BaseModel):
    model_path: str
    engine: str = ""
    use_azure: bool = False
    timeout: int = 60
    temperature: float = 0.0

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            openai.api_key = info["key"]
            self.engine = info["engine"]

    def run(self, prompt: str) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                key = "engine" if self.use_azure else "model"
                kwargs = {key: self.engine}
                response = openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    request_timeout=self.timeout,
                    temperature=self.temperature,  # this is the degree of randomness of the model's output
                    **kwargs,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content
            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("OpenAIModel request failed, retrying.")

        return output


class Scorer(BaseModel):
    @staticmethod
    def run(pred: str, gold: str) -> float:
        pred = pred.replace("$", "")
        gold = gold.replace("$", "")
        return float(pred.startswith(gold))


def evaluate(
    model: OpenAIModel,
    data_train: ReasonData,
    data_test: ReasonData,
    prompter: Prompter,
    file: TextIO,
) -> dict:
    is_correct = []
    score = 0

    progress = tqdm(data_test.samples)
    sample: ReasonSample

    for sample in progress:
        prompt = prompter.run(data_train, sample)
        raw = model.run(prompt).strip()
        sample.pred = prompter.get_answer(raw)
        is_correct.append(Scorer.run(sample.pred, sample.answer))
        score = sum(is_correct) / len(is_correct)
        progress.set_postfix(score=score)
        print(dict(prompt=prompt, raw=raw, gold=sample.answer, pred=sample.pred))
        print(sample.json(), file=file)

    return dict(score=score)


def main(
    data_name: str,
    prompt_name: str,
    data_path: str = None,
    num_shots: int = 4,
    max_test_samples: int = 500,
    seed: int = 0,
    **kwargs,
):
    model = OpenAIModel(**kwargs)
    print(locals())

    data_train = ReasonData.load_from_name(data_name, split="train", sample=True)
    name = f"{prompt_name}_{num_shots=}_{max_test_samples=}"
    data_test = ReasonData.load_from_name(data_name, split="test")

    path_out = f"outputs/{data_name}/{kwargs.get('model_name')}/{name}.json"
    if num_shots >= 0:
        data_train.samples = data_train.samples[:num_shots]
    if 0 < max_test_samples < len(data_test.samples):
        random.seed(seed)
        data_test.samples = random.sample(data_test.samples, k=max_test_samples)
    print(dict(train=len(data_train.samples), test=len(data_test.samples)))

    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    with open(path_out, "w") as f:
        prompter = select_prompter(prompt_name)
        result = evaluate(model, data_train, data_test, prompter, file=f)

    print(path_out)
    print(result)
    return result["score"]


if __name__ == "__main__":
    Fire()
