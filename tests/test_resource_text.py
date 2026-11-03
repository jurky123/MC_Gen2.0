import pytest
from mcgen.resource_text import extract_resource_text as extract

@pytest.mark.parametrize('filename,expected',[
 ('sun_gem.png','sun gem'),('soul_crystal.png','soul crystal'),('glowing_sapphire_hoe.png','glowing sapphire hoe'),
 ('rabbage.png','rabbage'),('TitanBase_sword.png','titan base sword'),('jade_gem.png','jade gem'),
 ('cactus_bow_pulling_0.png','cactus bow'),('oak_bow_pulling_2_spruce_arrow.png','oak bow spruce arrow'),
 ('gem4.png','gem 4'),('tier2_sword.png','tier 2 sword'),('item.sun_gem.png','sun gem'),
 ('botania:mana_diamond.png','mana diamond'),('blue_helmet_frame_2.png','blue helmet')])
def test_preserves_semantics_and_removes_only_explicit_technical_state(filename,expected):
 assert extract(filename,mod='jade')['text']==expected

@pytest.mark.parametrize('filename,mod,paths',[
 ('body_tinkers_thinking_burnt_ashes.png','',[]),('shield_materialis_alfsteel.png','',[]),
 ('sensing_spell_book_e.png','',[]),('spruce_sword_blade.png','',[]),('sbse7.png','',[]),
 ('fiery_ingot.png','',['assets/tetra/textures/item/module/sword/pommel/grip_loop/fiery_ingot.png']),
 ('skull.png','',['assets/elixirum/textures/item/elixir/cap/skull.png'])])
def test_components_layers_and_codes_never_become_whole_item_prompts(filename,mod,paths):
 r=extract(filename,mod,paths);assert r['decision']=='quarantine' and r['text'] is None

def test_whole_helmet_is_not_rejected_by_prior_visual_caption():
 assert extract('shadow_helmet.png')['text']=='shadow helmet'

def test_sparse_mask_never_becomes_ingot():
 assert extract('fiery_ingot.png',visible_pixels=3)['text'] is None
