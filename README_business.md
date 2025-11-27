## L1BSR – Satellite super‑resolution that turns open data into decision‑grade insights

**L1BSR** is a Sentinel‑2 image super‑resolution technology, originating from an award‑winning research project (**Best Student Paper – EarthVision 2023**).  
It produces **5 m high‑resolution** images from raw Sentinel‑2 L1B data at **10 m**, while **accurately re‑aligning spectral bands** – and does so **without HR ground‑truth data** (self‑supervised learning).

This repository hosts the reference PyTorch implementation, a public dataset and an online demo.

---

### Business perspective

- **Problem we address**  
  Decision‑makers in **agriculture, environment, urban planning, insurance, defense**, and others need finer analysis from optical satellite imagery. The main bottleneck is no longer data access, but **spatial resolution** and **radiometric quality** of the freely available imagery (e.g., Sentinel‑2).

- **Value proposition**  
  **L1BSR** increases the business value of Sentinel‑2 by:
  - **doubling spatial resolution** (from 10 m to 5 m) on BGRN bands,
  - **precisely aligning spectral bands**, which greatly improves the reliability of multispectral indicators,
  - relying on **self‑supervised learning**, meaning **no annotation cost and no need for proprietary HR data**.

- **Impact for verticals**  
  - **Precision agriculture**: better detection of intra‑parcel variability, finer monitoring of water and nutrient stress.
  - **Environment & climate**: more accurate mapping of wetlands, shorelines, forests and water surfaces.
  - **Urban & infrastructure**: clearer reading of fine structures (secondary roads, small buildings, industrial footprints).
  - **Risk & insurance**: improved detection of events (floods, fires, landslides) using open satellite data.

---

### Why this technology is different

- **Backed by state‑of‑the‑art research**  
  - Method published at **CVPR / EarthVision 2023**, awarded **Best Student Paper**.  
  - Public IPOL demo and open dataset on Zenodo, ensuring **transparency** and **reproducibility**.

- **Self‑supervised on real data**  
  - No need for synthetic LR/HR pairs or expensive high‑resolution sensors.  
  - Smart use of **Sentinel‑2 L1B detector overlap** to “create” supervision directly from the sensor.

- **Drop‑in for existing pipelines**  
  - Input: a single Sentinel‑2 L1B image.  
  - Output: a 5 m HR image with properly aligned bands, ready to plug into existing analytics pipelines (indices, segmentation, object detection, etc.).

---

### What this repository contains

- **Reference PyTorch implementation**  
  - **REC** (REConstruction) module: super‑resolution + band alignment.  
  - **CSR** (Cross‑Spectral Registration) module: dense motion fields between spectral bands.

- **Associated resources**  
  - Link to the scientific paper (arXiv).  
  - Link to the **L1BSR dataset** (Zenodo).  
  - Link to the **interactive demo** (IPOL).  
  - Example test images in `examples/`.

- **Quick local demo**  
  - `main.py` lets you run:
    - **super‑resolution** (x2) on an example image,  
    - **cross‑spectral registration** between bands.

---

### What this project shows about my profile

This project showcases my expertise in:

- **Deep learning for Earth Observation**
  - Super‑resolution and cross‑spectral registration for Sentinel‑2.
  - Training and optimizing models on large volumes of real satellite data.

- **Scientific & industrial computer vision**
  - Bridging academic research (EarthVision / IPOL) and real‑world use cases (agriculture, environment, risk).
  - Designing robust, explainable and maintainable vision pipelines.

- **Data & MLOps mindset**
  - Handling large‑scale geospatial datasets and preprocessing pipelines.
  - Preparing models for industrialization (serving, monitoring, scaling).

---

### My role on this project

On this project, I:

- Contributed to the **PyTorch implementation** of the L1BSR architecture, in particular the REC (super‑resolution) and CSR (cross‑spectral registration) modules.  
- Ran experiments on real Sentinel‑2 data to validate both the super‑resolution performance and the quality of cross‑spectral registration.  
- Helped structure the repository and documentation so that the method can be used and reproduced by practitioners (data scientists, ML engineers, EO specialists).  

---

### Tech stack & key skills

- **Core technologies**
  - `Python`, `PyTorch`  
  - Image processing (`numpy`, `scipy`, geospatial libraries upstream/downstream)  
  - GPU infrastructure (CUDA)

- **Key skills**
  - Deep learning modeling (super‑resolution, registration, differentiable warping)  
  - Optimizing models on noisy / real‑world data  
  - Reading, understanding and implementing research papers

---

- ### How to try it quickly

- **Prerequisites**
  - `Python 3`  
  - CUDA‑enabled GPU recommended for reasonable inference times.

- **Steps (high‑level)**
  - Clone the repository and install dependencies from `requirements.txt`.  
  - Use `main.py` with the sample images provided in `examples/`.  
  - Try the two main tasks:
    - **Super‑resolution**: generate a 5 m HR image.  
    - **Cross‑spectral registration**: align B, R, NIR bands to the G band.

For detailed technical usage (CLI arguments, figures, equations), refer to the main `README.md` and the scientific paper.

---

### Learn more / get in touch

- **Scientific & technical**
  - Paper: see the “paper” badge in the main `README.md`.  
  - Demo: see the “IPOL Demo” badge.

- **Opportunities & collaborations**
  - If you would like to discuss collaboration, internships, PhD topics or roles around these themes, you can:
    - explore this repository and the associated paper,  
    - prepare a few questions and potential use‑case ideas,  
    - and reach out via the profiles listed in the main `README.md` or your usual contact.

L1BSR is a demonstrator of what we can do in satellite super‑resolution. We are looking for people who want to **build the next bricks**: more generic models, new sensors, new data products.
