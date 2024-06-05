# ComicOCR
please remember to go install tesseract.exe
```bash
pip install -r requirements.txt
```
to get text and pos by tesseract ocr
```python
import usecomicsocr
print(usecomicsocr.textnpos(image_path="path/to/image", show=True, log=True))
```
to get ai corrected version of text
```python
import gui
print(gui.create_dialogue_options("path/to/image", "path/to/txt/to/save dialogue"))
```
