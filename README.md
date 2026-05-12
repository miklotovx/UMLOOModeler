# Reference Implementations  
**Paper:** Documenting AI Systems under the EU AI Act: A UML Architectural Framework with Support for Post-Hoc XAI

Link: https://zenodo.org/records/19599421

---

## Overview

This repository contains the **reference implementations used in the paper**.

The purpose of these materials is **illustrative and documentary**. They demonstrate how heterogeneous AI systems and post-hoc explainability methods can be structured according to the proposed architectural contract, enabling their representation through UML and automated extraction via UMLOOModeler.

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

  - CNN + DeepLift
  - Transformer + Integrated Gradients

---

## Scope and Limitations

- These examples are **not reference architectures** and are **not intended for benchmarking**.  
- Model performance and explanation quality are secondary to **architectural clarity and traceability**.  
- The code is intentionally structured to make architectural roles explicit, supporting traceability and auditability.

---

## About UMLOOModeler

This repository provides materials demonstrating *how the proposed framework can be instantiated in practice*.

UMLOOModeler tool is available at: https://umloomodeler.streamlit.app/

Other details about it (design goals, usage, and roadmap) are available at:  
https://github.com/miklotovx/UMLOOModeler/discussions/1

---

## License

All materials available in this repository are under the CC BY 4.0 license. You are free to use this material, including for commercial purposes, as long you cite the author.
