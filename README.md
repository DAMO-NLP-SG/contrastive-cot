# Contrastive Chain-of-Thought Prompting

[![AP](https://img.shields.io/badge/arXiv-Preprint-blue)](https://arxiv.org/abs/2311.09277)

This repository implements our work on [Contrastive Chain-of-Thought Prompting](https://arxiv.org/abs/2311.09277).

![intro](assets/ContrastiveIntro.png)

### Abstract

Despite the success of chain of thought in enhancing language model reasoning, the underlying process remains less well
understood. Although logically sound reasoning appears inherently crucial for chain of thought, prior studies
surprisingly reveal minimal impact when using invalid demonstrations instead. Furthermore, the conventional chain of
thought does not inform language models on what mistakes to avoid, which potentially leads to more errors. Hence,
inspired by how humans can learn from both positive and negative examples, we propose contrastive chain of thought to
enhance language model reasoning. Compared to the conventional chain of thought, our approach provides both valid and
invalid reasoning demonstrations, to guide the model to reason step-by-step while reducing reasoning mistakes. To
improve generalization, we introduce an automatic method to construct contrastive demonstrations. Our experiments on
reasoning benchmarks demonstrate that contrastive chain of thought can serve as a general enhancement of
chain-of-thought prompting.

### Setup

```
conda create -n contrastive-cot python=3.10 -y
conda activate contrastive-cot
pip install -r requirements.txt
```

### API Credentials

Update your [OpenAI key](https://platform.openai.com/account/api-keys) in [openai_info.json](openai_info.json):

```
{
  "engine": "gpt-3.5-turbo-0301",
  "key": "YOUR API KEY"
}
```

### Simple Inference

```
python demo.py infer contrast_cot --model_name openai --model_path openai_info.json \
--question "Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?"

{
  "question": "Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?",
  "prompt": "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nExplanation: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted.\nAnswer: 6\nWrong explanation: There are 21 - 15 = 6 trees originally. Then there were 15 trees after the Grove workers planted some more. So there must have been 21 trees that were planted.\nWrong Answer: 21\n\nQuestion: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot??\nExplanation: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot.\nAnswer: 5\nWrong explanation: There are originally 3 + 2 = 5 cars. Then 3 more cars arrive. Now 2 cars are in the parking lot\nWrong Answer: 2\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nExplanation: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total.\nAnswer: 39\nWrong explanation: Originally, Leah had 32 + 42 = 74 chocolates and her sister had 32. So in total they had 74 - 35 = 39. After eating 35, they had 42 pieces left in total\nWrong Answer: 42\n\nQuestion: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nExplanation: Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.\nAnswer: 8\nWrong explanation: Jason had 20 - 12 = 8 lollipops originally. Then he had 20 after giving some to Denny. So he gave Denny 12 lollipops\nWrong Answer: 12\n\nQuestion: Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?\nExplanation: ",
  "output": "Henry traveled a total of 60 miles on his bike trip. His second stop was 15 miles before the end of the trip, so he had already traveled 60 - 15 = 45 miles. His first stop was after 20 miles, so he traveled 45 - 20 = 25 miles between his first and second stops.\nAnswer: 25\nWrong explanation: Henry traveled a total of 60 miles on his bike trip. His first stop was after 20 miles, so he traveled 20 miles between his starting point and his first stop. His second stop was 15 miles before the end of the trip, so he traveled 60 - 15 = 45 miles between his second stop and the end of the trip. Therefore, he traveled a total of 20 + 45 = 65 miles between his first and second stops.\nWrong Answer: 65"
}
```

### Evaluation

Evaluate GSM8K with chain-of-thought

```
python demo.py main gsm8k cot --model_name openai --model_path openai_info.json
{'score': 0.692}
```

Evaluate GSM8K with contrastive chain-of-thought
(In this demo we use rationales manually written with incoherent bridging objects
by [Wang et al., 2023](https://aclanthology.org/2023.acl-long.153/))

```
python demo.py main gsm8k contrast_cot --model_name openai --model_path openai_info.json
{'score': 0.796}
```

(Note that some variance is possible)

### Reference

```
@misc{chia2023contrastive,
    title={Contrastive Chain-of-Thought Prompting},
    author={Yew Ken Chia and Guizhen Chen and Luu Anh Tuan and Soujanya Poria and Lidong Bing},
    year={2023},
    eprint={2311.09277},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```