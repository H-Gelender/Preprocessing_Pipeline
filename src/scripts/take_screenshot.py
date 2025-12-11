
from html2image import Html2Image
import time
import os

def capture():
    # Configure generic chrome path incase standard one isn't found
    # In debian container: /usr/bin/chromium
    hti = Html2Image(output_path='/app/output', custom_flags=['--no-sandbox', '--disable-gpu'])
    hti.browser_executable = '/usr/bin/chromium'
    
    print("Capturing Dashboard...")
    # Capture Navigator
    hti.screenshot(url='http://127.0.0.1:8080', save_as='dashboard_navigator.png', size=(1280, 720))
    print("Saved dashboard_navigator.png")

if __name__ == "__main__":
    if not os.path.exists('/usr/bin/chromium'):
        print("Error: /usr/bin/chromium not found")
    else:
        try:
            capture()
        except Exception as e:
            print(f"Capture failed: {e}")
