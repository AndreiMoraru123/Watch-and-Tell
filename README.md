# Watch & Tell

## [Show and Tell](https://arxiv.org/abs/1411.4555), but on video

### This project has been completely reworked into an improved version and is now obsolete.
### [Join the dark side here on the new repository](https://github.com/AndreiMoraru123/ContextCollector)

![pred](https://user-images.githubusercontent.com/81184255/197354068-834c3258-b953-4cf0-b1c2-9ffa6d26d020.gif)

Here is a quick Jupyter mini guide on how to use the Python COCO Api: [PythonAPI.pdf](https://github.com/AndreiMoraru123/Watch-and-Tell/files/9844733/PythonAPI.pdf)

```bash
pip install pycocotools-windows
```

Run the [make](https://github.com/AndreiMoraru123/Watch-and-Tell/blob/main/make.sh) script to get the [COCO dataset](https://cocodataset.org/#home) (50 GB, 2017 challenge) (requires gnu [wget](https://www.gnu.org/software/wget/))

#### Download the YOLO weights (I have included the class names and config file here, but these are too big)

```bash
cd YOLO
wget https://pjreddie.com/media/files/yolov3.weights
```

Train:

```bash
python train.py
```

Run:

```bash
python run.py
```

Based on the original architecture (and repo), this is using ResNet-152 as the encoder, and the LSTM as the decoder


CNN Encoder          |  RNN Decoder
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/81184255/197387871-4396b61c-0de0-433e-93b3-7fc3dedb1f8a.png)| ![image](https://user-images.githubusercontent.com/81184255/197387930-68f0a256-572f-42b1-9b93-45068740aa88.png)

Using [Darknet's](https://pjreddie.com/darknet/yolo/) YOLO to constrain where the model should look

```bibtex
@misc{https://doi.org/10.48550/arxiv.1411.4555,
  doi = {10.48550/ARXIV.1411.4555},
  url = {https://arxiv.org/abs/1411.4555},
  author = {Vinyals, Oriol and Toshev, Alexander and Bengio, Samy and Erhan, Dumitru},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Show and Tell: A Neural Image Caption Generator},
  publisher = {arXiv},
  year = {2014},
  copyright = {arXiv.org perpetual, non-exclusive lic
```

### To be improved: :heavy_check_mark: (Visit the new [repo](https://github.com/AndreiMoraru123/ContextCollector))

- [x] Migrate to OpenCV GPU build
- [x] Add an attention mechanism to the Decoder
- [x] Optimize model parameter size for inference speed
- [x] Change the greedy nearest word search to a beam search for the words in the vocabulary
