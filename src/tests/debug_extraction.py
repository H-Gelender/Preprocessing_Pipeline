
import os
import sys
import json

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph.builder import UMLSEnricher

def debug_extraction():
    print("\n--- Debugging Extraction Logic ---")
    
    # Real text from patch_1070
    roi_vlm = {
      "tissue_architecture": "The tissue architecture appears disorganized with a mix of dense and loose areas, suggesting a possible pathological process.",
      "glandular_structures": "No distinct glandular structures are visible in the provided image.",
      "stromal_components": "The stroma appears to be composed of a dense infiltrate of inflammatory cells with areas of fibrosis.",
      "inflammation": "There is a significant inflammatory infiltrate present, primarily composed of lymphocytes and other immune cells.",
      "pathological_patterns": "The image shows a pattern suggestive of chronic inflammation or an inflammatory condition, with no clear evidence of neoplastic changes."
    }
    
    # Init with temp cache
    cache_dir = '../output/TENX96/cache'
    enricher = UMLSEnricher(cache_dir)
    
    print("Testing extraction on ROI VLM data...")
    concepts = enricher.extract_concepts(roi_vlm)
    
    print(f"\nExtracted {len(concepts)} concepts:")
    for c in concepts:
        print(f" - {c}")
        
    if not concepts:
        print("\n❌ FAILED. No concepts extracted.")
        # Debug parsing
        print("\nDebugging Parsing Steps:")
        roi_keys = ['tissue_architecture', 'stromal_components', 'inflammation', 'pathological_patterns', 'cell_morphology']
        for k in roi_keys:
            if k in roi_vlm:
                text = roi_vlm[k]
                words = [w.strip(".,;:()").lower() for w in text.split()]
                stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'is', 'are', 'with', 'to', 'from', 'by', 'appear', 'appears', 'showing', 'suggesting', 'presence'}
                filtered = [w for w in words if w not in stopwords and len(w) > 3]
                print(f"Key '{k}' -> Candidates: {filtered}")
                
                # Check API for first few
                for cand in filtered[:3]:
                    res = enricher.search_term(cand)
                    print(f"  Query '{cand}' -> {res}")

if __name__ == "__main__":
    debug_extraction()
