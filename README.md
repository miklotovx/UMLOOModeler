# Reference Implementations - Paper: *Documenting AI Systems under the EU AI Act: A UML Framework for Post-Hoc XAI Compliance*
Paper is available at: https://zenodo.org/records/18404982
This directory contains the **reference implementations used in the paper**.
The purpose of these materials is **illustrative and documentary**.  
They demonstrate how heterogeneous AI systems and post-hoc explainability mechanisms can be modeled using the proposed UML-based framework and automatically documented with UMLOOModeler.

## What is included

For each module presented in the paper, this repository provides:

-the **Python source code** used in the illustrative examples  
-the **expected runtime outputs** (console logs and XAI visualizations)  
-the **UML class diagrams automatically generated** by UMLOOModeler from the source code  

These artifacts allow readers to directly verify the correspondence between:
implementation => UML extraction => compliance-oriented documentation.

## Modules (directories)

-**ClinicalModule**  
  Tabular data models using:
  
  -MLP + LIME  
  -Random Forest + SHAP  

-**ImageModule**  
  Image-based classification using:
  
  -CNN + LIME  

-**GeneticModule**  
  Sequential genomic data using:
  
  -BiLSTM + DeepSHAP  

## Scope and limitations

-The examples are **not reference architectures** and **not intended for performance benchmarking**.

-Model accuracy and explanation quality are secondary to **architectural clarity and traceability**.

-The code is structured to make architectural roles explicit for UML extraction and auditability.

## About UMLOOModeler

This repository documents *how the framework is instantiated in practice*.  
Details about the UMLOOModeler tool itself (design goals, usage, and roadmap) are available at: https://github.com/miklotovx/UMLOOModeler/discussions/1
