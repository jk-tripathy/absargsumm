import pytest

from utils import GSumGuidance, GSumOracleSummary


@pytest.fixture
def raw_src():
    return "How are you ? I am fine , thank you . And you ? I am fine too . thank you ."


@pytest.fixture
def raw_tgt():
    return "How are you ? I am fine"


@pytest.fixture
def src():
    return "How are you ?<q>I am fine , thank you .<q>And you ?<q>I am fine too .<q>thank you ."


@pytest.fixture
def tgt():
    return "How are you ?<q>I am fine"


@pytest.fixture
def oracle_summary():
    return "How are you ?<q>I am fine too ."


@pytest.fixture
def guidance_sentence():
    return "fine"


def test_format_text(raw_src, src, raw_tgt, tgt):
    guidance = GSumGuidance()
    formatted_src, formatted_tgt = guidance.format_text(raw_src, raw_tgt)
    assert formatted_src == src
    assert formatted_tgt == tgt


def test_get_oracle(src, tgt, oracle_summary):
    oracle = GSumOracleSummary()
    sentence = oracle.get_oracle(src, tgt)
    assert sentence == oracle_summary


def test_get_guidance(src, tgt, guidance_sentence):
    guidance = GSumGuidance()
    output = guidance.get_guidance(src, tgt)
    assert output == guidance_sentence
