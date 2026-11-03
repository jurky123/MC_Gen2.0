"""Keep filename supervision when an optional descriptive caption is unusable."""
from .semantic_captions import filename_words

REJECT_FLAGS = {'review_schema_error', 'name_conflict', 'background_in_caption'}


def caption_supervision(filename, descriptive_text, quality_flags=(), review_reason=None):
    short = ' '.join(filename_words(filename))
    if not short:
        raise ValueError('Filename provides no usable text supervision')
    reason = review_reason
    if not reason and set(quality_flags) & REJECT_FLAGS:
        reason = 'annotation_flags:' + ','.join(sorted(set(quality_flags) & REJECT_FLAGS))
    if not reason and (not isinstance(descriptive_text, str) or not descriptive_text.strip()):
        reason = 'missing_descriptive_text'
    if not reason and '_' in descriptive_text:
        reason = 'identifier_separators_in_description'
    long = descriptive_text.strip() if not reason else None
    return {'filename_text': short, 'descriptive_text': long,
            'long_caption_eligible': long is not None,
            'long_caption_rejection': reason,
            'caption_variants': ['short', 'long'] if long is not None and long != short else ['short']}
