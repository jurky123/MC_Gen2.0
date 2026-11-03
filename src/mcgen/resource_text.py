"""Deterministic file-information supervision; no model-generated text or visual judgments."""
import re
from pathlib import PurePosixPath

POLICY_VERSION='resource_text_v1'
FORMS=set('pickaxe sword shovel axe hoe helmet chestplate leggings boots shield ingot nugget gem crystal apple book bow arrow dust wand staff potion bottle bucket ring dye food seed rod stick'.split())
TECHNICAL_NAMESPACES={'tconstruct','etshtinker','plusticreforged','tinkersreforged','tinkersadvanced'}

def tokens(text):
 text=re.sub(r'([A-Z]+)([A-Z][a-z])',r'\1 \2',text)
 text=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',text)
 return re.findall(r'[a-z]+|[0-9]+',text.lower())

def extract_resource_text(filename,mod='',paths=(),visible_pixels=None):
 basename=PurePosixPath(str(filename).replace('\\','/')).name
 stem=re.sub(r'\.png$','',basename,flags=re.I);original=tokens(stem);words=list(original);removed=[];reasons=[]
 context=(stem+' '+mod).lower();tinker=any(x in context for x in ('tinker','tconstruct','materialis','plustic','construct','tcintegrations'))
 # Paths are recovered by exact basename + pixels upstream, never guessed from a material name.
 for path in paths:
  p='/'+path.lower()
  if any(x in p for x in ('/module/','/tool/cleaver/','/trims/','/elixir/cap/','/elixir/shape/')):reasons.append('resource_component')
  if any(x in p for x in ('/overlay/','/overlays/','/gui/','/hud/')):reasons.append('resource_layer')
 if re.search(r'_(?:e|n|s)$',stem,re.I):reasons.append('possible_emissive_normal_or_specular_map')
 w=set(words)
 if w & {'overlay','underlay','hud','cursor','slot','placeholder','uvmap','texturemap'}:reasons.append('layer_or_ui_identifier')
 if w & {'binding','toolrod','toolhead','crossguard','embellishment','embellishments','grip','guard'}:reasons.append('assembly_component')
 if ('head' in w and w & {'pickaxe','axe','shovel','hoe','hammer','mattock','scythe'}) or ('sword' in w and 'blade' in w):reasons.append('tool_component')
 if tinker and w & {'body','shield','limb','plating','front','back','broken','handle','blade','pick'}:reasons.append('modular_tool_component')
 if w & {'part','parts'} and w & FORMS:reasons.append('item_component')
 if 'model' in w or 'png' in w:reasons.append('model_or_nested_export_identifier')
 if visible_pixels is not None and visible_pixels<8:reasons.append('insufficient_visible_content')
 if re.fullmatch(r'[a-zA-Z]{1,6}[0-9]+',stem) and not w & FORMS:reasons.append('opaque_code_identifier')
 if not any(x.isalpha() for x in words):reasons.append('empty_semantic_name')
 # Drop only explicit resource namespaces; never strip a material because a mod shares its name.
 if ':' in stem:
  namespace=stem.split(':',1)[0];removed+=tokens(namespace);words=tokens(stem.split(':',1)[1])
 if words and words[0]=='item':removed.append(words.pop(0))
 kept=[]
 for word in words:
  if word in TECHNICAL_NAMESPACES:removed.append(word)
  else:kept.append(word)
 words=kept
 # The bow state index is technical. Keep accompanying arrow/material names.
 if 'bow' in words and 'pulling' in words:
  i=words.index('pulling');removed.append(words.pop(i))
  if i<len(words) and words[i] in {'0','1','2'}:removed.append(words.pop(i))
 for marker in ['frame','anim']:
  if marker in words:
   i=words.index(marker)
   if i+1<len(words) and words[i+1].isdigit():removed.extend(words[i:i+2]);del words[i:i+2]
 text=' '.join(words)
 if not text or not any(c.isalpha() for c in text):reasons.append('empty_semantic_name')
 return {'policy_version':POLICY_VERSION,'text':text if not reasons else None,'candidate_text':text,
         'original_tokens':original,'removed_technical_tokens':removed,'resource_paths':list(paths),
         'reasons':sorted(set(reasons)),'decision':'quarantine' if reasons else 'keep',
         'provenance':'resource_file_information_only','path_verified':bool(paths)}
