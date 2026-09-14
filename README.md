# Reference Implementations and UMLOOModeler Documentation

**Paper:** Documenting AI Systems under the EU AI Act: A UML Architectural Framework with Support for Post-Hoc XAI

[Paper available on Zenodo](https://zenodo.org/records/19599421)

---

## Overview

This repository provides materials demonstrating *how the proposed framework in the paper can be instantiated in practice*. It also provides the technical documentation of UMLOOModeler and an informal report explaining how that documentation was produced with human guidance and LLM support.

These examples serve as reusable templates for developing more sophisticated post-hoc XAI systems with different data modalities, predictive models, and explanation techniques.

---

## Contents

### Reference implementations

For each module presented in the paper, the repository provides:

- Python source code used in the examples;
- expected runtime outputs;
- UML class diagram images automatically generated from the source code with UMLOOModeler.

These materials allow readers to examine the correspondence between:

> implementation → UML extraction → Documentation

The `ClinicalModule` directory also contains a traceable technical documentation example based on the `clinical_shap.py` implementation and the evidence produced by UMLOOModeler:

- [Clinical SHAP example documentation](ClinicalModule/clinical_shap_example_documentation.pdf)

### UMLOOModeler documentation

The `UMLOOModeler Documentation` directory contains:

- [UMLOOModeler Technical Documentation](<UMLOOModeler Documentation/UMLOOModeler_Technical_Documentation.pdf>) - describes the behavior, supported contracts, structural evidence, outputs, and known limitations of UMLOOModeler 1.1;
- [How the Documentation Was Produced](<UMLOOModeler Documentation/How_the_documentation_was_produced.pdf>) - reports the progressive human-LLM process used to produce and validate the technical documentation.

The second document is an informal experience report. It reconstructs the process observed while documenting UMLOOModeler and does not present a formally validated methodology.

---

## Modules

### ClinicalModule

Tabular data examples using:

- MLP + LIME;
- Random Forest + SHAP.

### ImageModule

Image classification example using:

- CNN + LIME.

### GeneticModule

Sequential genomic data example using:

- BiLSTM + DeepSHAP.

### Other examples

Additional examples using:

- Autoencoder + Occlusion;
- CNN + DeepLIFT;
- Gradient Boosting + ICE;
- GRU + GSHAP;
- ResNet18 + Grad-CAM;
- Random Forest + Ceteris;
- Transformer + Integrated Gradients;
- ViT + Integrated Gradients.

---

## Scope of the Examples

- These examples are **not intended for benchmarking**.
- Model performance and explanation quality are secondary to **architectural clarity and traceability**.
- The code is intentionally structured to make architectural roles explicit, supporting traceability and auditability.

---

## About UMLOOModeler

The UMLOOModeler web is available [here](https://umloomodeler.streamlit.app/).

Offline editions for Windows and Linux are available on the [GitHub Releases](https://github.com/miklotovx/UMLOOModeler/releases) page. The offline editions process source code locally and are recommended for sensitive or confidential codebases.

---

## License

All materials available directly in this repository, such as documentation, examples, and public project materials, are licensed under the CC BY 4.0 license. You are free to use these repository materials, including for commercial purposes, as long as you cite the author.

The UMLOOModeler offline distributions available through GitHub Releases are licensed separately. The Community Edition is provided for non-commercial use only and is governed by the `LICENSE.txt` file included in each release package.

---

## Contact

You can find my email address in the paper.
