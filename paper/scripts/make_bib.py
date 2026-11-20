"""Build refs.bib for splncs04 from the project's paper list (data/refs_src.bib) plus classic references.

- arXiv-only entries become @article{journal = arXiv preprint arXiv:<id>} so splncs04 prints them properly;
- entries whose peer-reviewed version is known are rewritten with that venue;
- author lists longer than three names are cut to "first three and others" (printed as et al.);
- arXiv DOIs are dropped (redundant for preprints, wrong for published versions).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "data" / "refs_src.bib").read_text(encoding="utf-8")

VENUE = {  # key -> (type, {field: value})
    "Jaume2019FUNSD": ("inproceedings", {"booktitle": "2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)", "pages": "1--6", "year": "2019"}),
    "Huang2021SROIE": ("inproceedings", {"booktitle": "2019 International Conference on Document Analysis and Recognition (ICDAR)", "pages": "1516--1520", "year": "2019"}),
    "Liu2023OCRBench": ("article", {"journal": "Science China Information Sciences", "volume": "67", "number": "12", "pages": "220102", "year": "2024"}),
    "Ouyang2024OmniDocBench": ("inproceedings", {"booktitle": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", "year": "2025"}),
    "Romero2024CVQA": ("inproceedings", {"booktitle": "Advances in Neural Information Processing Systems 37 (NeurIPS), Datasets and Benchmarks Track", "year": "2024"}),
    "Kim2021Donut": ("inproceedings", {"booktitle": "Computer Vision -- ECCV 2022", "series": "LNCS", "volume": "13688", "pages": "498--517", "publisher": "Springer", "year": "2022"}),
    "Sclar2023FormatSpread": ("inproceedings", {"booktitle": "International Conference on Learning Representations (ICLR)", "year": "2024"}),
    "Gebru2018Datasheets": ("article", {"journal": "Communications of the ACM", "volume": "64", "number": "12", "pages": "86--92", "year": "2021"}),
    "Kwon2023vLLMPagedAtt": ("inproceedings", {"booktitle": "Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)", "pages": "611--626", "year": "2023"}),
    "Huang2022LayoutLMv3": ("inproceedings", {"booktitle": "Proceedings of the 30th ACM International Conference on Multimedia", "pages": "4083--4091", "year": "2022"}),
}

EXTRA = r"""
@inproceedings{vu2021mcocr,
  title = {{MC-OCR} Challenge: Mobile-Captured Image Document Recognition for {V}ietnamese Receipts},
  author = {Vu, Xuan-Son and Bui, Quang-Anh and Nguyen, Nhu-Van and others},
  booktitle = {2021 RIVF International Conference on Computing and Communication Technologies (RIVF)},
  pages = {1--6}, year = {2021}
}
@inproceedings{nguyen2021vintext,
  title = {Dictionary-Guided Scene Text Recognition},
  author = {Nguyen, Nguyen and Nguyen, Thu and Tran, Vinh and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {7383--7392}, year = {2021}
}
@inproceedings{berzak2016anchoring,
  title = {Anchoring and Agreement in Syntactic Annotations},
  author = {Berzak, Yevgeni and Huang, Yan and others},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages = {2215--2224}, year = {2016}
}
@inproceedings{park2019cord,
  title = {{CORD}: A Consolidated Receipt Dataset for Post-{OCR} Parsing},
  author = {Park, Seunghyun and Shin, Seung and Lee, Bado and others},
  booktitle = {Workshop on Document Intelligence at NeurIPS 2019},
  year = {2019}
}
@inproceedings{mathew2021docvqa,
  title = {{DocVQA}: A Dataset for {VQA} on Document Images},
  author = {Mathew, Minesh and Karatzas, Dimosthenis and Jawahar, C. V.},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages = {2200--2209},
  year = {2021}
}
@inproceedings{singh2019textvqa,
  title = {Towards {VQA} Models That Can Read},
  author = {Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {8317--8326},
  year = {2019}
}
@article{cohen1960kappa,
  title = {A Coefficient of Agreement for Nominal Scales},
  author = {Cohen, Jacob},
  journal = {Educational and Psychological Measurement},
  volume = {20}, number = {1}, pages = {37--46}, year = {1960}
}
@article{scott1955pi,
  title = {Reliability of Content Analysis: The Case of Nominal Scale Coding},
  author = {Scott, William A.},
  journal = {Public Opinion Quarterly},
  volume = {19}, number = {3}, pages = {321--325}, year = {1955}
}
@article{fleiss1971kappa,
  title = {Measuring Nominal Scale Agreement Among Many Raters},
  author = {Fleiss, Joseph L.},
  journal = {Psychological Bulletin},
  volume = {76}, number = {5}, pages = {378--382}, year = {1971}
}
@article{landis1977agreement,
  title = {The Measurement of Observer Agreement for Categorical Data},
  author = {Landis, J. Richard and Koch, Gary G.},
  journal = {Biometrics},
  volume = {33}, number = {1}, pages = {159--174}, year = {1977}
}
@book{krippendorff2018content,
  title = {Content Analysis: An Introduction to Its Methodology},
  author = {Krippendorff, Klaus},
  edition = {4th}, publisher = {SAGE}, address = {Thousand Oaks, CA}, year = {2018}
}
@article{hayes2007alpha,
  title = {Answering the Call for a Standard Reliability Measure for Coding Data},
  author = {Hayes, Andrew F. and Krippendorff, Klaus},
  journal = {Communication Methods and Measures},
  volume = {1}, number = {1}, pages = {77--89}, year = {2007}
}
@article{kuhn1955hungarian,
  title = {The {H}ungarian Method for the Assignment Problem},
  author = {Kuhn, Harold W.},
  journal = {Naval Research Logistics Quarterly},
  volume = {2}, number = {1--2}, pages = {83--97}, year = {1955}
}
@book{efron1993bootstrap,
  title = {An Introduction to the Bootstrap},
  author = {Efron, Bradley and Tibshirani, Robert J.},
  publisher = {Chapman \& Hall/CRC}, address = {New York}, year = {1993}
}
@book{corbin2015basics,
  title = {Basics of Qualitative Research: Techniques and Procedures for Developing Grounded Theory},
  author = {Corbin, Juliet and Strauss, Anselm},
  edition = {4th}, publisher = {SAGE}, address = {Thousand Oaks, CA}, year = {2015}
}
@book{creswell2018mixed,
  title = {Designing and Conducting Mixed Methods Research},
  author = {Creswell, John W. and Plano Clark, Vicki L.},
  edition = {3rd}, publisher = {SAGE}, address = {Thousand Oaks, CA}, year = {2018}
}
@article{oconnor2020icr,
  title = {Intercoder Reliability in Qualitative Research: Debates and Practical Guidelines},
  author = {O'Connor, Cliodhna and Joffe, Helene},
  journal = {International Journal of Qualitative Methods},
  volume = {19}, pages = {1--13}, year = {2020}
}
"""

out = []
for ent in re.split(r"\n(?=@)", src.strip()):
    m = re.match(r"@(\w+)\{([^,]+),", ent)
    if not m:
        continue
    typ, key = m.group(1).lower(), m.group(2).strip()
    fields = dict(re.findall(r"^\s*(\w+)\s*=\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\"[^\"]*\"|\d+)\s*,?\s*$", ent, flags=re.M))
    fields = {k.lower(): v.strip() for k, v in fields.items()}
    for k in ("note", "archiveprefix", "editor", "month", "address", "url", "abstract"):
        fields.pop(k, None)
    # arXiv DOIs duplicate the arXiv id (preprints) or point to the wrong version (published papers)
    if "10.48550" in fields.get("doi", ""):
        fields.pop("doi")
    # shorten author lists
    au = fields.get("author", "")
    au_inner = au[1:-1] if au[:1] in "{\"" else au
    names = [a.strip() for a in re.split(r"\s+and\s+", au_inner.replace("\n", " ")) if a.strip()]
    if len(names) > 3:
        au_inner = " and ".join(names[:3]) + " and others"
    fields["author"] = "{" + au_inner + "}"
    if key in VENUE:
        typ, upd = VENUE[key]
        fields.pop("eprint", None)
        fields.pop("journal", None)
        for k, v in upd.items():
            fields[k] = "{" + v + "}"
    elif "eprint" in fields and typ == "misc":
        typ = "article"
        eid = fields.pop("eprint").strip("{}\"")
        fields["journal"] = "{arXiv preprint arXiv:" + eid + "}"
    else:
        fields.pop("eprint", None)
    body = ",\n".join(f"  {k} = {v}" for k, v in fields.items())
    out.append(f"@{typ}{{{key},\n{body}\n}}")

models_tbd = r"""
@misc{qwen35,
  title = {{Qwen3.5} Model Card},
  author = {{Qwen Team}},
  howpublished = {\url{https://huggingface.co/Qwen}},
  year = {2026}
}
@article{ministral3,
  title = {{Ministral 3}},
  author = {Liu, Alexander H. and Khandelwal, Kartik and Subramanian, Sandeep and others},
  journal = {arXiv preprint arXiv:2601.08584},
  year = {2026}
}
@article{mimovl2025,
  title = {{MiMo-VL} Technical Report},
  author = {{Xiaomi LLM-Core Team}},
  journal = {arXiv preprint arXiv:2506.03569},
  year = {2025}
}
@misc{erax2024,
  title = {{EraX-VL-7B-V1.5} Model Card},
  author = {{EraX AI}},
  howpublished = {\url{https://huggingface.co/erax-ai/EraX-VL-7B-V1.5}},
  year = {2024}
}
"""
(ROOT / "refs.bib").write_text("\n\n".join(out) + "\n" + EXTRA + models_tbd, encoding="utf-8")
print("entries:", len(out) + EXTRA.count("\n@") + models_tbd.count("\n@"))
