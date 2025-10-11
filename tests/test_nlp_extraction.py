import pathlib
import sys
import types


class _TorchStub(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _NoGradContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    @staticmethod
    def no_grad():
        return _TorchStub._NoGradContext()


class _TransformersStub(types.SimpleNamespace):
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, *_args, **_kwargs):
            return {}

    class AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def __call__(self, *_args, **_kwargs):
            class _Score:
                def __init__(self, value: float) -> None:
                    self._value = value

                def item(self) -> float:
                    return self._value

            class _Logits:
                def softmax(self, *_inner, **_kwargs):
                    return [_Score(0.0), _Score(0.0), _Score(0.0)]

            return types.SimpleNamespace(logits=_Logits())


class _StanzaStub(types.SimpleNamespace):
    @staticmethod
    def Pipeline(*_args, **_kwargs):
        raise RuntimeError("stanza.Pipeline should not be invoked in unit tests")

    @staticmethod
    def download(*_args, **_kwargs):
        return None


sys.modules.setdefault("torch", _TorchStub())
sys.modules.setdefault("transformers", _TransformersStub())
sys.modules.setdefault("stanza", _StanzaStub())

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_extraction import EntityRelationExtractor, ExtractionConfig


def test_extract_filters_generic_spans(monkeypatch):
    config = ExtractionConfig(
        ner_exclude_terms=set(),
        generic_span_blocklist={"adverse effect", "brain organoid"},
    )
    extractor = EntityRelationExtractor(config=config, stanza_pipelines=[])

    generic_entities = [
        {
            "start": None,
            "end": None,
            "entity_group": "PROBLEM",
            "word": "adverse effects",
            "lemma": "adverse effects",
            "tokens": ["adverse", "effects"],
            "upos": ["ADJ", "NOUN"],
            "xpos": ["JJ", "NNS"],
            "source_package": "tokenize=default, ner=i2b2",
            "coref_antecedent": None,
        },
        {
            "start": None,
            "end": None,
            "entity_group": "CELL_TYPE",
            "word": "brain organoids",
            "lemma": "brain organoids",
            "tokens": ["brain", "organoids"],
            "upos": ["NOUN", "NOUN"],
            "xpos": ["NN", "NNS"],
            "source_package": "tokenize=default, ner=jnlpba",
            "coref_antecedent": None,
        },
    ]

    informative_entity = {
        "start": None,
        "end": None,
        "entity_group": "PROBLEM",
        "word": "insomnia",
        "lemma": "insomnia",
        "tokens": ["insomnia"],
        "upos": ["NOUN"],
        "xpos": ["NN"],
        "source_package": "tokenize=default, ner=i2b2",
        "coref_antecedent": None,
    }

    extractor._ner_pipeline = lambda _text: [*generic_entities, informative_entity]
    extractor.classify_relation_via_nli = lambda _c, _s, _o: (None, None)

    text = (
        "Reports discussed insomnia along with adverse effects and findings in brain "
        "organoids."
    )
    extraction = extractor.extract({"id": "paper1", "title": "paper1"}, text)
    assert extraction is not None

    node_names = [node.canonical_name for node in extraction.nodes]
    assert node_names == ["Insomnia"]
