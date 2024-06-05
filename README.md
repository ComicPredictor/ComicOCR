# ComicOCR
(fork of [largecats/comics-ocr](https://github.com/largecats/comics-ocr))

please remember to go install tesseract.exe and
```bash
pip install -r requirements.txt
```
to get text and pos by tesseract ocr
```python
import usecomicsocr
print(usecomicsocr.textnpos(image_path="path/to/image", show=True, log=True))
```
to get LLM corrected version of text
```python
import gui
print(gui.create_dialogue_options("path/to/image", "path/to/txt/to/save/dialogue"))
```
