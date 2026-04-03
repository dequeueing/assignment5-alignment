import json
from pathlib import Path


test_path = "/home/exouser/cs336/assignment5-alignment/data/gsm8k/test.jsonl"
train_path = "/home/exouser/cs336/assignment5-alignment/data/gsm8k/train.jsonl"



# TODO: the original format:
# {"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18"}
# target: the answer part should be like this:

"<think>Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n</think> <answer> 18 </answer> "


# 也就是说，我希望达成的效果是增加两个 tag，一个是 think，一个是 answer。

# 你可以检查一下，每一个 JSON 文件里面的 item 其实都有非常类似的格式：
# 1. 开头有四个井号（####）
# 2. 后面跟着 question
# 3. answer 在末尾，也有四个井号

# 我需要把它们 format 一下，让 answer 的格式符合我要进行 fine-tuning 的数据格式。

# 另外需要提醒你的是，这些文件的规模都很大，基本都是几千行的规模。所以如果你直接读取一个文件，很可能会把你的 context window 直接耗光。请你一定要切记，不要直接读取整个文件，好吗？


def split_answer(raw_answer: str) -> tuple[str, str]:
	"""Split GSM8K answer into rationale(thinking) and final short answer.

	Expected raw format usually ends with: "...\n#### 18".
	"""
	marker = "####"
	marker_idx = raw_answer.rfind(marker)
	if marker_idx == -1:
		# Fallback: keep all content in think and leave final answer empty.
		return raw_answer.strip(), ""

	think = raw_answer[:marker_idx].rstrip()
	final_answer = raw_answer[marker_idx + len(marker) :].strip()
	return think, final_answer


def format_question_and_answer(question: str, raw_answer: str) -> tuple[str, str, str]:
	think, final_answer = split_answer(raw_answer)
	formatted_question = f"{question.strip()} <think>"
	formatted_answer = f"{think}</think> <answer> {final_answer} </answer>"
	return formatted_question, formatted_answer, final_answer


def reformat_jsonl_file(input_path: str) -> Path:
	src = Path(input_path)
	dst = src.with_suffix(".formatted.jsonl")

	with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
		for line_no, line in enumerate(fin, start=1):
			line = line.strip()
			if not line:
				continue

			try:
				item = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON at {src}:{line_no}: {exc}") from exc

			if "question" not in item:
				raise KeyError(f"Missing 'question' field at {src}:{line_no}")

			if "answer" not in item:
				raise KeyError(f"Missing 'answer' field at {src}:{line_no}")

			formatted_question, formatted_answer, final_answer = format_question_and_answer(
				str(item["question"]),
				str(item["answer"]),
			)
			item["question"] = formatted_question
			item["answer"] = formatted_answer
			item["ground truth"] = final_answer
			fout.write(json.dumps(item, ensure_ascii=False) + "\n")

	return dst


def main() -> None:
	for path in [train_path, test_path]:
		output_path = reformat_jsonl_file(path)
		print(f"Done: {path} -> {output_path}")


if __name__ == "__main__":
	main()