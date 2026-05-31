from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "True if the answer is grounded in the facts"
    ]


CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def format_document_context(documents: list[dict]) -> str:
    parts = []
    for document in documents or []:
        text = document.get("text") or document.get("page_content") or ""
        text = str(text).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _score_bool(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


class AnswerJudge:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self.model = model
        self._correctness_llm = ChatOpenAI(
            model=model, temperature=temperature
        ).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)
        self._relevance_llm = ChatOpenAI(
            model=model, temperature=temperature
        ).with_structured_output(RelevanceGrade, method="json_schema", strict=True)
        self._grounded_llm = ChatOpenAI(
            model=model, temperature=temperature
        ).with_structured_output(GroundedGrade, method="json_schema", strict=True)

    def judge_correctness(
        self,
        *,
        question: str,
        gold_answer: str,
        student_answer: str,
    ) -> float:
        prompt = (
            f"QUESTION: {question}\n"
            f"GROUND TRUTH ANSWER: {gold_answer}\n"
            f"STUDENT ANSWER: {student_answer}"
        )
        grade = self._correctness_llm.invoke(
            [
                {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        return _score_bool(grade["correct"]) or 0.0

    def judge_relevance(self, *, question: str, student_answer: str) -> float:
        prompt = f"QUESTION: {question}\nSTUDENT ANSWER: {student_answer}"
        grade = self._relevance_llm.invoke(
            [
                {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        return _score_bool(grade["relevant"]) or 0.0

    def judge_groundedness(
        self,
        *,
        documents: list[dict],
        student_answer: str,
    ) -> float:
        facts = format_document_context(documents)
        prompt = f"FACTS: {facts}\nSTUDENT ANSWER: {student_answer}"
        grade = self._grounded_llm.invoke(
            [
                {"role": "system", "content": GROUNDED_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        return _score_bool(grade["grounded"]) or 0.0


def evaluate_answer_record(
    record: dict,
    judge: AnswerJudge,
    *,
    metrics: list[str] | None = None,
) -> dict:
    metrics = metrics or ["correctness", "groundedness", "relevance"]
    question = record.get("question", "")
    student_answer = record.get("answer", "")
    gold_answer = record.get("gold_answer", "")
    documents = record.get("documents") or []

    result = {"qa_index": record.get("qa_index")}
    if record.get("error"):
        for name in metrics:
            result[name] = None
        return result

    if "correctness" in metrics:
        result["correctness"] = judge.judge_correctness(
            question=question,
            gold_answer=gold_answer,
            student_answer=student_answer,
        )
    if "groundedness" in metrics:
        result["groundedness"] = judge.judge_groundedness(
            documents=documents,
            student_answer=student_answer,
        )
    if "relevance" in metrics:
        result["relevance"] = judge.judge_relevance(
            question=question,
            student_answer=student_answer,
        )
    return result


def aggregate_answer_metrics(per_query: list[dict], metrics: list[str]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    if not per_query:
        return summary

    for name in metrics:
        values = [row[name] for row in per_query if row.get(name) is not None]
        summary[name] = sum(values) / len(values) if values else None
    return summary


def merge_retrieval_and_answer_metrics(
    retrieval_rows: list[dict],
    answer_rows: list[dict],
) -> list[dict]:
    answer_by_index = {
        row.get("qa_index"): row for row in answer_rows if row.get("qa_index") is not None
    }
    merged = []
    for row in retrieval_rows:
        combined = dict(row)
        answer_row = answer_by_index.get(row.get("qa_index"), {})
        for key in ("correctness", "groundedness", "relevance"):
            if key in answer_row:
                combined[key] = answer_row[key]
        merged.append(combined)
    return merged


def compute_e2e_success_rate(
    per_query: list[dict],
    *,
    rule: str = "correctness_and_groundedness_pass",
) -> float | None:
    if not per_query:
        return None

    passed = 0
    for row in per_query:
        if rule == "correctness_and_groundedness_pass":
            if row.get("correctness") == 1.0 and row.get("groundedness") == 1.0:
                passed += 1
        else:
            if row.get("correctness") == 1.0:
                passed += 1
    return passed / len(per_query)
