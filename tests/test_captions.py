import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mcgen.captions import parse_candidate, CAPTION_INSTRUCTION


def example():
    return dict(object='sword',colors=['gray','brown'],short_caption='A gray sword with a brown handle.',
                detailed_caption='A diagonal gray blade with a brown handle and a short crossguard.',
                certainty='clear',uncertainty='')


def test_self_confidence_never_approves_training():
    r=parse_candidate('```json\n'+json.dumps(example())+'\n```')
    assert r['status']=='candidate_needs_visual_review'
    assert r['approved_for_training'] is False


def test_uncertain_or_invented_claims_are_quarantined():
    a=example();a['certainty']='ambiguous';a['object']='unknown'
    assert 'uncertain_identity' in parse_candidate(json.dumps(a))['quality_flags']
    a=example();a['short_caption']='A legendary sword used for mining.'
    assert 'unsupported_function_or_lore' in parse_candidate(json.dumps(a))['quality_flags']


def test_schema_and_budget_are_strict():
    a=example();a['short_caption']='word '*17
    r=parse_candidate(json.dumps(a))
    assert r['annotation']['short_caption']==' '.join(['word']*17)
    assert 'short_caption_long' in r['format_warnings']
    a=example();a['confidence']=1.
    with pytest.raises(ValueError,match='schema'):parse_candidate(json.dumps(a))
    with pytest.raises(ValueError,match='JSON'):parse_candidate('not JSON')
    assert 'filename' not in json.dumps(example())


def test_extra_colors_do_not_erase_valid_caption():
    a=example();a['colors']=['red','blue','green','white','black']
    r=parse_candidate(json.dumps(a))
    assert r['annotation']['short_caption']==a['short_caption']
    assert r['format_warnings']==['many_colors']
