from rag.analyzer import (
    CHUNKING_MISS,
    HALLUCINATION,
    RETRIEVAL_MISS,
    SUPPORTED,
    UNKNOWN,
    TraceAnalyzer,
)
from rag.judge import JudgeResult, _parse_judge_output
from utils.text_utils import split_into_sentences


class FakeEmbedder:
    keys = ["paris", "france", "capital", "banana", "founded", "1961", "mars"]

    def embed_text(self, text: str):
        lowered = text.lower()
        return [float(lowered.count(key)) for key in self.keys]

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]


class FakeJudge:
    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        claim_l = claim.lower()
        evidence_l = evidence.lower()

        if "mars" in claim_l or "banana" in claim_l:
            return JudgeResult("unsupported", "not in evidence")

        if "paris" in claim_l or "capital of france" in claim_l:
            if "paris" in evidence_l and "capital" in evidence_l and "france" in evidence_l:
                return JudgeResult("supported", "entailed by evidence")
            return JudgeResult("unsupported", "paris fact missing")

        if "founded in 1961" in claim_l:
            if "founded" in evidence_l and "1961" in evidence_l:
                return JudgeResult("supported", "both halves present")
            if "founded" in evidence_l or "1961" in evidence_l:
                return JudgeResult("partial", "only one half")
            return JudgeResult("unsupported", "cern fact missing")

        return JudgeResult("unsupported", "default")


def _analyzer():
    return TraceAnalyzer(FakeEmbedder(), FakeJudge())


def test_supported_when_retrieved_entails_claim():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
        corpus_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
    )
    assert report.sentences[0].fault == SUPPORTED
    assert report.summary[SUPPORTED] == 1


def test_retrieval_miss_when_corpus_has_evidence_but_retriever_did_not():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[{"id": "r1", "text": "Bananas are a yellow fruit."}],
        corpus_chunks=[
            {"id": "r1", "text": "Bananas are a yellow fruit."},
            {"id": "c9", "text": "Paris is the capital of France."},
        ],
    )
    assert report.sentences[0].fault == RETRIEVAL_MISS
    assert report.sentences[0].component == RETRIEVAL_MISS
    assert report.sentences[0].evidence_rank is None
    assert report.root_cause == RETRIEVAL_MISS
    assert "c9" in report.sentences[0].supporting_chunk_ids


def test_hallucination_when_neither_retrieved_nor_corpus_support_claim():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Bananas grow on Mars.",
        retrieved_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
        corpus_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
    )
    assert report.sentences[0].fault == HALLUCINATION


def test_loose_judge_does_not_treat_a_fragment_as_full_support():
    class LooseJudge:
        def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
            claim_tokens = {word.lower() for word in claim.split() if len(word) > 2}
            evidence_tokens = {word.lower() for word in evidence.split() if len(word) > 2}
            if claim_tokens & evidence_tokens or (claim_tokens and claim_tokens <= evidence_tokens):
                return JudgeResult("supported", "any overlap")
            return JudgeResult("unsupported", "no overlap")

    report = TraceAnalyzer(FakeEmbedder(), LooseJudge()).analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[
            {"id": "r1", "text": "Paris is the capital"},
            {"id": "r2", "text": "of France."},
        ],
        corpus_chunks=[
            {"id": "r1", "text": "Paris is the capital"},
            {"id": "r2", "text": "of France."},
        ],
    )
    assert report.sentences[0].fault == CHUNKING_MISS


def test_year_split_is_chunking_miss_even_when_most_words_are_in_one_chunk():
    class LooseJudge:
        def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
            claim_tokens = {word.lower().strip(".,") for word in claim.split() if len(word) > 2}
            evidence_tokens = {word.lower().strip(".,") for word in evidence.split() if len(word) > 2}
            if claim_tokens & evidence_tokens:
                return JudgeResult("supported", "any overlap")
            return JudgeResult("unsupported", "no overlap")

    report = TraceAnalyzer(FakeEmbedder(), LooseJudge()).analyze(
        question="When did the Oak Street library open?",
        answer="The Oak Street library opened in 1998.",
        retrieved_chunks=[
            {"id": "left", "text": "The Oak Street library opened"},
            {"id": "right", "text": "in 1998."},
        ],
        corpus_chunks=[
            {"id": "left", "text": "The Oak Street library opened"},
            {"id": "right", "text": "in 1998."},
        ],
    )
    assert report.sentences[0].fault == CHUNKING_MISS


def test_chunking_miss_when_only_concatenated_chunks_entail_claim():
    report = _analyzer().analyze(
        question="When was CERN founded?",
        answer="CERN was founded in 1961.",
        retrieved_chunks=[
            {"id": "r1", "text": "CERN was founded"},
            {"id": "r2", "text": "in 1961 near Geneva."},
        ],
        corpus_chunks=[
            {"id": "r1", "text": "CERN was founded"},
            {"id": "r2", "text": "in 1961 near Geneva."},
        ],
    )
    assert report.sentences[0].fault == CHUNKING_MISS


def test_missing_evidence_pool_is_unknown_rather_than_a_guess():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[{"id": "r1", "text": "Bananas are a yellow fruit."}],
    )
    assert report.sentences[0].fault == UNKNOWN
    assert report.sentences[0].component == UNKNOWN
    assert any("unknown" in note for note in report.notes)


def test_logged_rank_names_k_too_small():
    from rag.analyzer import K_TOO_SMALL

    fillers = [{"id": f"d{i}", "text": f"Bananas are a yellow fruit {i}."} for i in range(5)]
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=fillers,
        corpus_chunks=fillers + [{"id": "c9", "text": "Paris is the capital of France.", "rank": 8, "score": 0.2}],
    )
    assert report.sentences[0].fault == RETRIEVAL_MISS
    assert report.sentences[0].component == K_TOO_SMALL
    assert report.sentences[0].evidence_rank == 8


def test_logged_query_rewrite_is_its_own_stage():
    from rag.analyzer import QUERY_REWRITE_FAILURE

    report = _analyzer().analyze(
        question="bananas",
        answer="Paris is the capital of France.",
        retrieved_chunks=[{"id": "r1", "text": "Bananas are a yellow fruit."}],
        corpus_chunks=[{"id": "c9", "text": "Paris is the capital of France.", "rank": 1, "score": 0.4}],
        original_query="What is the capital of France?",
        rewritten_query="bananas",
    )
    assert report.sentences[0].fault == QUERY_REWRITE_FAILURE
    assert report.sentences[0].component == QUERY_REWRITE_FAILURE


def test_parse_judge_json_and_fenced_output():
    parsed = _parse_judge_output('{"label":"supported","reason":"yes"}')
    assert parsed.label == "supported"
    fenced = _parse_judge_output('```json\n{"label": "partial", "reason": "half"}\n```')
    assert fenced.label == "partial"
    bad = _parse_judge_output("not json")
    assert bad.label == "unsupported"


def test_sentence_split_handles_missing_space_after_period():
    text = (
        "Necrotizing fasciitis is caused by bacteria in the vast majority of cases, "
        "though fungi can also rarely lead to this condition as well."
        "Many cases of necrotizing fasciitis are caused by group A beta-hemolytic streptococci."
    )
    assert split_into_sentences(text) == [
        "Necrotizing fasciitis is caused by bacteria in the vast majority of cases, "
        "though fungi can also rarely lead to this condition as well.",
        "Many cases of necrotizing fasciitis are caused by group A beta-hemolytic streptococci.",
    ]
