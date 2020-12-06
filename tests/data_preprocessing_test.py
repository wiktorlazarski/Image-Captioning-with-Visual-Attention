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


def test_idx2word_raises_index_error(vocabulary: dp.Vocabulary) -> None:
    # given
    index = -1
    expected_message = "No word is mapped to -1"

    # when / then
    with pytest.raises(IndexError) as result:
        vocabulary.idx2word(index)

    assert str(result.value) == expected_message


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
