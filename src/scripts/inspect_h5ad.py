
import scanpy as sc

path = '/app/hest_data/st/TENX96.h5ad'
adata = sc.read_h5ad(path)
print("Adata Shape:", adata.shape)
print("Var Columns:", adata.var.columns.tolist())
print("Uns Keys:", adata.uns.keys())

if 'highly_variable' in adata.var.columns:
    print("Found 'highly_variable' genes:", adata.var['highly_variable'].sum())

# Check for any other boolean flags
for col in adata.var.columns:
    if adata.var[col].dtype == bool:
        print(f"Bool Col '{col}': {adata.var[col].sum()} True")
