import pytest
from mcgen.caption_supervision import caption_supervision

@pytest.mark.parametrize('caption,flags,reason', [
    (None, [], None), ('', [], None), ('a_sun_gem', [], None),
    ('A red sword.', ['name_conflict'], None),
    ('A glowing sun gem.', [], 'Unsupported glowing inference'),
])
def test_rejected_description_retains_filename(caption, flags, reason):
    result = caption_supervision('sun_gem.png', caption, flags, reason)
    assert result['filename_text'] == 'sun gem'
    assert result['descriptive_text'] is None
    assert result['caption_variants'] == ['short']
    assert result['long_caption_rejection']

def test_uncertain_name_does_not_drop_good_description():
    result = caption_supervision('sun_gem.png', 'A sun gem with a yellow body.', ['name_uncertain'])
    assert result['caption_variants'] == ['short', 'long']
    assert result['long_caption_rejection'] is None

def test_identical_caption_is_not_duplicated():
    assert caption_supervision('sun_gem.png', 'sun gem')['caption_variants'] == ['short']

def test_empty_filename_cannot_create_empty_training_prompt():
    with pytest.raises(ValueError):
        caption_supervision('_.png', None)
