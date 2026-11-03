"""Check the portable public subset without downloading the original full dataset."""
import json
from pathlib import Path
from PIL import Image
from mcgen.catalog import file_hash
from mcgen.images import content_hash
from mcgen.resource_text import extract_resource_text
if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]/'examples/data_review';rows=json.load(open(root/'samples.json'))
 for r in rows:
  image=root/r['image'];assert file_hash(image)==r['image_sha256'];assert Image.open(image).size==(16,16);assert content_hash(Image.open(image))==r['target_hash']
  assert extract_resource_text(**r['file_information'])==r['recorded_result'],r['id']
  assert (root/r['source']['license_file']).is_file()
 for r in json.load(open(root/'text_cases.json')):assert extract_resource_text(r['filename'])==r['recorded_result'],r['filename']
 issues=[r['id'] for r in rows if r['human_review']['status']=='known_component_omission']
 print(f'PASS: {len(rows)} images and12 filename cases reproduce current rules. Known semantic omissions: {", ".join(issues)}. This does not certify semantic correctness.')
