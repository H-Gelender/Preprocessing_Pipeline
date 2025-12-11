
import os
import requests
from dotenv import load_dotenv

# Load env
load_dotenv('/app/.env')
api_key = os.getenv('UMLS_API_KEY')

if not api_key:
    print("❌ UMLS_API_KEY not found in .env")
    exit(1)

print(f"✅ Found API Key: {api_key[:4]}...{api_key[-4:]}")

# Test Query
base_uri = 'https://uts-ws.nlm.nih.gov/rest'
version = 'current'
endpoint = '/search/current'
query = {'string': 'carcinoma', 'apiKey': api_key, 'rootSource': 'MTH'}

try:
    r = requests.get(f"{base_uri}{endpoint}", params=query)
    r.raise_for_status()
    data = r.json()
    
    results = data['result']['results']
    print(f"✅ Query successful! Found {len(results)} results for 'carcinoma'.")
    print(f"Top Result: {results[0]['name']} (CUI: {results[0]['ui']})")

except Exception as e:
    print(f"❌ API Request Failed: {e}")
    if 'r' in locals():
        print(r.text)
