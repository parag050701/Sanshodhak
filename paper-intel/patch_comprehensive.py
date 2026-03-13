import re

with open("/home/admin-/Desktop/Sanshodhak/paper-intel/run_comprehensive_eval.py", "r") as f:
    content = f.read()

# Replace imports to add scipy
new_imports = """import json
import time
import os
import requests
import uuid
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import math
import numpy as np
from collections import defaultdict
from sentence_transformers import CrossEncoder
from scipy.stats import ttest_rel"""
content = re.sub(r"import json.*from collections import defaultdict", new_imports, content, flags=re.DOTALL)

# Add cross-encoder variant
variant_code = """
        # Test 5: HGR+CE (Hybrid + Graph + Cross-Encoder)
        self.logger._log("Evaluating variant: HGR+CE (Hybrid+Graph+CrossEncoder)", 'info')
        self.rag.config['enable_bm25'] = True
        self.rag.config['enable_graph'] = True
        hgr_ce_results = self._evaluate_variant("HGR+CE")
        all_results["HGR+CE"] = hgr_ce_results
"""
content = content.replace("all_results[\"HGR\"] = hgr_results", "all_results[\"HGR\"] = hgr_results\n" + variant_code)

# Add statistical testing
stat_code = """
        # Calculate Statistical Significance between VR-D and HGR
        self.logger._log("Calculating Statistical Significance (Paired t-test)...", 'info')
        if "VR-D" in all_results and "HGR" in all_results:
            vrd_ndcg = [q['ndcg_5'] for q in all_results['VR-D']['query_results']]
            hgr_ndcg = [q['ndcg_5'] for q in all_results['HGR']['query_results']]
            if len(vrd_ndcg) == len(hgr_ndcg) and len(vrd_ndcg) > 1:
                t_stat, p_val = ttest_rel(vrd_ndcg, hgr_ndcg)
                all_results['significance'] = {
                    'test': 'paired_t_test_ndcg_5',
                    'p_value': float(p_val),
                    'significant_at_05': bool(p_val < 0.05)
                }
"""
content = content.replace("return all_results", stat_code + "\n        return all_results")

# Write changes
with open("/home/admin-/Desktop/Sanshodhak/paper-intel/run_comprehensive_eval.py", "w") as f:
    f.write(content)
