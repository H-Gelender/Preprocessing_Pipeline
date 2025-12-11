
import os
import json
import sys

# Add parent dir to path to import builder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph.builder import UMLSEnricher, FolderManager

def test_enricher_logic():
    print("\n--- Testing UMLS Enricher Logic ---")
    # Mock VLM data
    vlm_data = {
        "cell_morphology": "Cells are irregular",
        "key_features": ["Carcinoma", "Mitosis", "Nucleus", "Stromal fibrosis"]
    }
    
    # Init Enricher
    # Use a temp cache dir for testing or the real one
    cache_dir = '../output/TENX96/cache' 
    if not os.path.exists(cache_dir):
        print(f"Warning: Cache dir {cache_dir} does not exist. Creating it.")
        os.makedirs(cache_dir, exist_ok=True)
        
    enricher = UMLSEnricher(cache_dir)
    
    print(f"Mock Input: {vlm_data['key_features']}")
    concepts = enricher.extract_concepts(vlm_data)
    print(f"Extracted Concepts: {concepts}")
    
    if not concepts:
        print("❌ Logic Test Failed or API Key missing/invalid.")
    else:
        print("✅ Logic Test Passed.")

def inspect_real_data():
    print("\n--- Inspecting Real KG Data ---")
    kg_dir = '../output/TENX96/knowledge_graph'
    
    if not os.path.exists(kg_dir):
        print(f"❌ KG Directory not found at {kg_dir}")
        return

    # Counter for stats
    total_patches = 0
    patches_with_vlm = 0
    patches_with_concepts = 0
    
    # Walk a bit
    for root, dirs, files in os.walk(kg_dir):
        if 'info.json' in files and 'patch_' in root:
            total_patches += 1
            with open(os.path.join(root, 'info.json'), 'r') as f:
                info = json.load(f)
            
            vlm = info.get('vlm_analysis')
            if vlm:
                patches_with_vlm += 1
                # Check concepts
                # Concepts might be in 'roi_context' -> 'patch_concepts'
                # or just 'umls_concepts' if we saved it directly?
                # The builder saves it into 'roi_context' for patches.
                
                ctx = info.get('roi_context', {})
                concepts = ctx.get('patch_concepts', [])
                if concepts:
                    patches_with_concepts += 1
                    if patches_with_concepts == 1:
                        print(f"✅ Found enriched patch at: {root}")
            
            if total_patches >= 200: break # Check more patches
            
    print(f"Checked {total_patches} patches.")
    print(f"Patches with VLM analysis: {patches_with_vlm}")
    print(f"Patches with UMLS Concepts: {patches_with_concepts}")
    
    if patches_with_vlm > 0 and patches_with_concepts == 0:
        print("⚠️ VLM data exists but UMLS concepts are missing. Did you run the 'enrich' step?")
        
        # Determine why it might be failing
        # Show a sample VLM output
        print("\nSample VLM Output from last patch:")
        if 'vlm' in locals() and vlm:
            print(json.dumps(vlm, indent=2))
    elif patches_with_vlm == 0:
        print("⚠️ No VLM data found. Did you run the VLM step?")

if __name__ == "__main__":
    test_enricher_logic()
    inspect_real_data()
