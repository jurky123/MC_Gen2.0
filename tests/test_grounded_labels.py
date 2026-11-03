from mcgen.grounded_labels import decide,resource_flags

def record(name='sun_gem.png',paths=(),pixels=60):
 return dict(source_filename=name,resource_paths=paths,visible_pixels=pixels,resource_flags=resource_flags(name,paths))
def visual(role='whole',caption='A yellow gem with a dark outline.'):
 return dict(object='gem',role=role,caption=caption,certainty='clear')
def semantic(name='sun gem',relation='compatible'):
 return dict(semantic_name=name,removed_tokens=[],relation=relation,reason='Visible gem identity agrees; sun is a name concept.')
def test_abstract_name_and_independent_visual_stay_separate():
 r=decide(record(),visual(),semantic());assert r['file_label']=='sun gem';assert r['visual_label']=='A yellow gem with a dark outline.'
def test_verified_component_overrides_hallucinated_whole_object():
 r=decide(record('fiery_ingot.png',['assets/tetra/textures/item/module/sword/pommel/grip_loop/fiery_ingot.png']),visual(caption='A solid metal bar.'),semantic('fiery ingot'))
 assert r['caption_variants']==[] and r['decision']=='quarantine'
def test_emissive_layer_is_not_a_book():
 r=record('book_e.png');r['resource_flags']=resource_flags('book_e.png',[], '_e')
 assert decide(r,visual(),semantic('book e'))['caption_variants']==[]
def test_dots_cannot_become_ingot():
 assert decide(record(pixels=3),visual(),semantic())['visual_label'] is None
def test_conflicting_name_does_not_replace_independent_visual():
 r=decide(record('unknown_resource.png'),visual(),semantic('unknown resource','conflict'));assert r['file_label'] is None;assert r['visual_label']
def test_invented_name_rejected():
 assert decide(record(),visual(),semantic('legendary sun gem'))['file_label'] is None
def test_silent_semantic_token_loss_rejected():
 assert decide(record(),visual(),semantic('gem'))['file_label'] is None
def test_component_by_vision_excludes_both_labels():
 assert decide(record(),visual('component'),semantic())['caption_variants']==[]
def test_visible_axe_and_hoe_name_require_review():
 v=visual();v['object']='axe'
 r=decide(record('sapphire_hoe.png'),v,semantic('sapphire hoe','uncertain'))
 assert r['caption_variants']==[] and 'visual_filename_identity_conflict' in r['quality_flags']
def test_rendering_comment_does_not_enter_training():
 assert decide(record(),visual(caption='A gem shown in two previews.'),semantic())['visual_label'] is None
def test_abstract_concept_cannot_be_removed_even_if_reported():
 s=semantic('gem');s['removed_tokens']=['sun']
 assert decide(record(),visual(),s)['file_label'] is None
def test_unfamiliar_name_is_retained_as_explicit_weak_semantics():
 assert decide(record('rabbage.png'),visual(),semantic('rabbage','uncertain'))['file_label']=='rabbage'
