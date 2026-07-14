# Reference Implementations  
**Paper:** Documenting AI Systems under the EU AI Act: A UML Architectural Framework with Support for Post-Hoc XAI

Link: https://zenodo.org/records/19599421

---

## Overview

This repository provides materials demonstrating *how the proposed framework in the paper can be instantiated in practice*.

These examples serve as reusable templates for developing more sophisticated post-hoc XAI systems with different data modalities, predictive models, and explanation techniques.

---

## Contents

For each module presented in the paper, the repository provides:

- Python source code used in the examples  
- Expected runtime outputs   
- UML class diagrams automatically generated from the source code with UMLOOModeler 

These materials allow readers to directly verify the correspondence between:

**implementation => UML extraction => compliance-oriented documentation**

---

## Modules (directories)

**ClinicalModule** Tabular data models using:
  
  - MLP + LIME  
  - Random Forest + SHAP  

**ImageModule** Image-based classification using:
  
  - CNN + LIME  

**GeneticModule** Sequential genomic data using:
  
  - Bilstm + DeepSHAP

**Other examples** Contains more examples:

  - Autoencoder + Occlusion
  - CNN + DeepLift
  - Gradient Boosting + Ice
  - GRU + GSHAP
  - ResNet18 + GradCam
  - Random Forest + Ceteris
  - Transformer + Integrated Gradients
  - Vit + Integrated Gradients

---

## Scope and Limitations

- These examples are **not intended for benchmarking**.  
- Model performance and explanation quality are secondary to **architectural clarity and traceability**.  
- The code is intentionally structured to make architectural roles explicit, supporting traceability and auditability.

---

## About UMLOOModeler

UMLOOModeler Web tool is available <a href="https://umloomodeler.streamlit.app/ target="_blank" rel="noopener noreferrer"> here </a>.

A standalone Windows version is available in the <a href="https://github.com/miklotovx/UMLOOModeler/releases" target="_blank" rel="noopener noreferrer">GitHub releases</a> page. A Linux version is also planned. The offline version is recommended for sensitive codes.

Other details about it (design goals, usage, and roadmap) are available at:  
https://github.com/miklotovx/UMLOOModeler/discussions/1

---

## License

All materials available directly in this repository, such as documentation, examples, and public project materials, are licensed under the CC BY 4.0 license. You are free to use these repository materials, including for commercial purposes, as long as you cite the author.

The UMLOOModeler desktop application distributed through GitHub Releases is licensed separately. The Community Edition is provided for non-commercial use only and is governed by the `LICENSE.txt` file included in the release package.
