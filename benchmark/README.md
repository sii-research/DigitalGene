# Digital Gene Benchmark – Evaluation

**TL;DR.** Given a **single-view image**, your model should output a **Digital Gene** (structural code) and a **3D object**. This repo provides a **unified evaluation pipeline** and scripts to score geometry similarity, pose consistency, and manipulability.

---

## What’s inside

* **Unified evaluation** for image → gene → 3D predictions.
* **Metrics covered**

  * **3D geometry similarity:** Chamfer Distance, F1@τ.
  * **Pose consistency:** absolute errors on **Yaw / Pitch / Roll**.
  * **VLM-based object similarity:** cosine similarity between visual-language embeddings of the rendered prediction and reference.
  * **Manipulability / axis metrics:** **Axis Angular Error** (deg) and **Axis Position Error** (distance).
* **Dataset adapters** for three evaluation sets (see below).

---

## Datasets

All evaluation data are hosted on Hugging Face:
**👉 [baibizhe/Digital\_Gene\_Benchmark](https://huggingface.co/datasets/baibizhe/Digital_Gene_Benchmark/tree/main)**

| Split               | Purpose                                                                      | Size / Notes                           |
| ------------------- | ---------------------------------------------------------------------------- | -------------------------------------- |
| **OmniObj-3D Set**  | Geometry & pose evaluation with paired (image ↔ 3D object).                  | **6813** image–mesh pairs.             |
| **Real Images Set** | Robustness on real photos; renders are compared via VLM similarity and pose. | **4000** real images.                  |
| **Operability Set** | Axis/affordance evaluation for objects with annotated rotation axes.         | **12** objects with ground-truth axes. |







---

## Metric definitions (brief)

* **Chamfer Distance (CD), F1@τ (surface overlap)**
Some evaluation code is from [1].  Corresponding dataset is in huggingface baibizhe/Digital_Gene_Benchmark/collected_pair_data_rotation.zip They are subset of the  OmniObject3D [2] .
```evaluating files are in evaluate_3Dmetrics_with_dataset_with_GTobj ```

* **Pose errors (Yaw / Pitch / Roll).**
We first sample surface points from the predicted and ground‑truth meshes, then perform a dense rotation grid search to coarsely align the prediction by minimizing symmetric Chamfer Distance. From the best‑matching rotation, we report the overall geodesic rotation angle and the absolute Euler errors (Yaw, Pitch, Roll) in degrees. Corresponding dataset is in huggingface baibizhe/Digital_Gene_Benchmark/collected_pair_data_rotation.zip. They are subset of the  OmniObject3D [2] .
```evaluating files are in evaluate_3Dmetrics_with_dataset_with_GTobj ```
* **VLM object similarity.**
We normalize the predicted mesh and render it from multiple canonical viewpoints. A visual‑language model is prompted to compare these renders with the reference image, focusing on 3D shape (ignoring color/texture), and to output a scalar **“FINAL SCORE: X.X”** in $[0,1]$. To improve stability, we repeat the query with random view subsets and report the mean of the valid scores. Corresponding real dataset is in huggingface baibizhe/Digital_Gene_Benchmark/3d_real_data.zip.zip. They collect by ourselves. 
```evaluating files are in evaluate_3Dmetrics_with_RealImages_WO_GTobj ```

* **Axis metrics (manipulability):**
  * **Axis Angular Error (deg):** angle between predicted and GT axis directions.
  * **Axis Position Error:** Euclidean distance between predicted and GT axis lines (e.g., shortest distance between lines or offset at reference contact point).
  Corresponding  dataset with object with axis is in huggingface baibizhe/Digital_Gene_Benchmark/paris_subset.zip. They are subset of the  Paris [3].
  Some part of the evaluating code is borrowed from  https://github.com/NVlabs/DigitalTwinArt/blob/1a48b402e4bf4bb7731296e8e230f0db3d86fe4f/eval/eval_results.py
```evaluating files are in evaluate_manipulate_with_rotAxis_diff_on_parisData ```


## Citation
[1] Huang, Z., Stojanov, S., Thai, A., Jampani, V., & Rehg, J.M. (2023). ZeroShape: Regression-Based Zero-Shot Shape Reconstruction. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10061-10071.

[2] Wu, T., Zhang, J., Fu, X., Wang, Y., Ren, J., Pan, L., Wu, W., Yang, L., Wang, J., Qian, C., Lin, D., & Liu, Z. (2023). OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 803-814.

[3] Liu, J., Mahdavi-Amiri, A., & Savva, M. (2023). PARIS: Part-level Reconstruction and Motion Analysis for Articulated Objects. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 352-363.
If this benchmark or the evaluation code is useful in your research, please cite:

```bibtex
@misc{digital_gene_benchmark,
  title  = {Digital Gene Benchmark: Single-View Image to Digital Gene and 3D Reconstruction},
  author = {B. Bai et al.},
  year   = {2025},
  note   = {Hugging Face dataset: baibizhe/Digital_Gene_Benchmark}
}
```

---

## License & Acknowledgements

* Code and evaluation scripts are released for research purposes.
* Datasets are hosted on Hugging Face under their respective licenses.
* Thanks to the contributors and maintainers of geometry processing, rendering, and VLM libraries used in this project.
