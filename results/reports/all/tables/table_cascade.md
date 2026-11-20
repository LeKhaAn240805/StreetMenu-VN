**Image to JSON against OCR text to JSON (test; Offer F1 %)**

| Model | P1 | OCR: paddlevl | OCR: ppocr | P-Oracle |
|---|---|---|---|---|
| Qwen3-8B (text) | – | 38.3 | 29.6 | 78.0 |
| VyLinh-3B (text) | – | 23.9 | 20.5 | 53.0 |

P1: the model reads the image. OCR: the model reads the transcript of the named OCR engine (plus the image for VLMs). P-Oracle: a human transcript of the item lines.
