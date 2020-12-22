import pytest
import scripts.data_processing as dp


@pytest.fixture
def vocabulary() -> dp.Vocabulary:
    return dp.Vocabulary()


@pytest.fixture
def text_pipeline() -> dp.TextPipeline:
    return dp.TextPipeline()


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


def test_word2idx_return_unknown_tag(vocabulary: dp.Vocabulary) -> None:
    # given
    word = "testnonexistingword"
    expected_result = vocabulary.word2idx("<UNK>")

    # when
    result = vocabulary.word2idx(word)

    # then
    assert result == expected_result


def test_text_pipeline_preprocessing(text_pipeline: dp.TextPipeline) -> None:
    # given
    text = "A on, of some-unknown."
    expected_result = [
        text_pipeline.vocabulary.word2idx("<SOS>"),
        text_pipeline.vocabulary.word2idx("a"),
        text_pipeline.vocabulary.word2idx("on"),
        text_pipeline.vocabulary.word2idx("of"),
        text_pipeline.vocabulary.word2idx("<UNK>"),
        text_pipeline.vocabulary.word2idx("<EOS>"),
    ]

    # when
    result = text_pipeline(text)

    # then
    assert result == expected_result


def test_pad_sequence(text_pipeline: dp.TextPipeline) -> None:
    # given
    pad_token_idx = text_pipeline.vocabulary.word2idx("<PAD>")
    captions_batch = [[10000, 38, 329, 0, 958, 10001], [10000, 38, 329, 10001]]
    expected_result = [
        [10000, 38, 329, 0, 958, 10001],
        [10000, 38, 329, 10001, pad_token_idx, pad_token_idx],
    ]

    # when
    result = text_pipeline.pad_sequences(captions_batch)

    # then
    assert result == expected_result
