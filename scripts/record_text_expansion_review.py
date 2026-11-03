"""Record all 21 candidate-sheet inspections, completed before expansion training."""
import json
from pathlib import Path
from mcgen.catalog import atomic_json,file_hash
ROOT=Path('outputs/text_expansion2k_20261102')
GROUPS={
'partial_asset_or_overlay':[75,315,368,403,560,562,582,591,666,766,809,836,921,922,1053,1080,1085,1086,1087,1088,1090,1091,1092,1098,1099,1100,1101,1102,1107,1108,1109,1110,1122,1123,1124,1125,1144,1149,1150,1152,1153,1154,1167,1168,1169,1170,1171,1172,1173,1174,1176,1178,1199,1268,1286,1327,1403,1421,1422,1454,1470,1471,1588,1589,1590,1591,1614,1647,1667,1738,1822,1828,1866,1907,1909,1917,1942,1944,1952],
'ui_mold_background_or_texture':[453,530,620,774,781,1083,1106,1790,1791,1923,1940,1980],
'different_or_ambiguous_category':[155,180,622,687,707,807,1198,1333,1359,1360,1369,1370,1435,1442,1448,1449,1455,1463,1474,1475,1484,1487,1490,1491,1492,1495,1496,1503,1509,1516,1519,1526,1527,1529,1530,1534,1535,1539,1555,1557,1558,1573,1734,1920,1921,1922,1956,1957,1962,1977],
}
if __name__=='__main__':
 rows=json.loads((ROOT/'candidates.json').read_text());reasons={i:reason for reason,indices in GROUPS.items() for i in indices}
 assert sum(map(len,GROUPS.values()))==len(reasons)
 atomic_json(ROOT/'full_visual_review.json',{'reviewer':'assistant_visual_review','scope':'All 1992 candidates inspected across page_00 through page_20 before expansion training. Review of image completeness and filename/category compatibility, NOT certification of all generated captions. Unusual complete mod designs retained; clear parts, layers, unrelated objects and UI assets excluded.','candidate_sha256':file_hash(ROOT/'candidates.json'),'reviewed_indices':list(range(len(rows))),'excluded':[{'id':r['id'],'index':r['review_index'],'reason':reasons[r['review_index']]} for r in rows if r['review_index'] in reasons],'kept':len(rows)-len(reasons)})
 print('visual exclusions',len(reasons))
