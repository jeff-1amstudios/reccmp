from unittest.mock import Mock
from reccmp.compare.db import EntityDb
from reccmp.compare.lines import LinesDb
from reccmp.compare.event import ReccmpReportProtocol
from .test_function_comparator import compare_functions

def test_effective_filtering_toggle():
    db = EntityDb()
    lines_db = LinesDb()
    report = Mock(spec=ReccmpReportProtocol)
    
    # cmp eax, ecx (39 c8) vs cmp ecx, eax (39 c1)
    # These are considered effective matches by patch_cmp_jmp
    orig = b"\x39\xc8"
    recm = b"\x39\xc1"
    
    # Verify that by default (show_all_diffs=False), the effective diff is filtered out
    res_filtered = compare_functions(db, lines_db, orig, recm, report, show_all_diffs=False)
    assert not any(tag in ("replace", "insert", "delete") for tag, _, _, _, _ in res_filtered.codes)
    
    # Verify that when show_all_diffs=True, the diff is preserved
    res_all = compare_functions(db, lines_db, orig, recm, report, show_all_diffs=True)
    assert any(tag == "replace" for tag, _, _, _, _ in res_all.codes)
