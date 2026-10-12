import os
import re
from playwright.sync_api import sync_playwright

def export_assets():
    html_path = os.path.abspath("workflow.html")
    file_url = f"file:///{html_path.replace(os.sep, '/')}"
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # High DPI for crisp rendering
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page.goto(file_url)
        page.wait_for_selector(".diagram-container svg")
        page.wait_for_timeout(1000) # Ensure fonts & layout settle
        
        # 1. Capture the diagram card as crisp PNG
        diagram_el = page.locator(".diagram-container")
        png_path = os.path.abspath("workflow.png")
        diagram_el.screenshot(path=png_path)
        print(f"[ok] Saved PNG to {png_path} ({os.path.getsize(png_path):,} bytes)")
        
        # 2. Extract SVG string via Archify serializer or SVG element
        svg_content = page.evaluate("""() => {
            const svg = document.querySelector('.diagram-container svg');
            const fontCss = document.getElementById('archify-fonts')?.textContent || '';
            const clonedSvg = svg.cloneNode(true);
            const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
            style.textContent = fontCss;
            clonedSvg.insertBefore(style, clonedSvg.firstChild);
            return '<?xml version="1.0" encoding="UTF-8"?>\\n' + new XMLSerializer().serializeToString(clonedSvg);
        }""")
        
        svg_path = os.path.abspath("workflow.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"[ok] Saved SVG to {svg_path} ({os.path.getsize(svg_path):,} bytes)")
        
        # 3. Test export download behavior to verify the file extension fix
        with page.expect_download() as download_info:
            page.evaluate("Archify.exportMenu.run('png')")
        download = download_info.value
        download_name = download.suggested_filename
        print(f"[ok] Browser Export download suggested filename: '{download_name}'")
        
        browser.close()

if __name__ == "__main__":
    export_assets()
