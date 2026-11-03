import json
from mcgen.caption_review import parse_review


def response(**changes):
    data=dict(semantic_name='packed_ice_sword',technical_tokens=[],visual_caption='A pale blue sword with a broad blade.',
              caption='A packed ice sword with a pale blue blade.',relation='compatible',reason='The visible sword matches the name.')
    return json.dumps(dict(data,**changes))


def test_readable_caption_and_name_preserved():
    r=parse_review(response(),'packed_ice_sword.png')
    assert r['combined_text']=='A packed ice sword with a pale blue blade.'
    assert r['effective_name']=='packed ice sword'
    assert not r['approved_for_training']


def test_fluent_caption_cannot_drop_semantic_name():
    r=parse_review(response(caption='A blue sword.'),'packed_ice_sword.png')
    assert r['status']=='quarantine'
    assert 'packed ice' in r['combined_text']
    assert 'caption_name_loss' in r['quality_flags']


def test_actual_conflict_remains_marked():
    r=parse_review(response(relation='conflict'),'packed_ice_sword.png')
    assert 'name_conflict' in r['quality_flags']
    assert r['proposed_views']==[]
