import pytest
import scripts.data_preprocessing as dp


@pytest.fixture
def vocabulary() -> dp.Vocabulary:
    return dp.Vocabulary()


def test_len(vocabulary: dp.Vocabulary) -> None:
    # given
    expected_result = 10_000 + len(vocabulary.SPECIAL_TOKENS)

    # when
    result = len(vocabulary)

    # then
    assert result == expected_result


def test_idx2word(vocabulary: dp.Vocabulary) -> None:
    # given
    index = 0
    expected_result = "a"

    # when
    result = vocabulary.idx2word(index)

    # then
    assert result == expected_result


def test_word2idx(vocabulary: dp.Vocabulary) -> None:
    # given
    word = "a"
    expected_result = 0

    # when
    result = vocabulary.word2idx(word)

    # then
    assert result == expected_result


def test_word2idx_UNK(vocabulary: dp.Vocabulary) -> None:
    # given
    word = "testnonexistingword"
    expected_result = vocabulary.word2idx("<UNK>")

    # when
    result = vocabulary.word2idx(word)

    # then
    assert result == expected_result
