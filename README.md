This repository contains the source code and an instruction how to use the tool for in-situ radiometric calibration of TLS. It accompanies the recent (submitted) publication:

# Automatic in-situ radiometric calibration of TLS: Compensating distance and angle of incidence effects using overlapping scans (submitted)

**[Helena Laasch](https://gseg.igp.ethz.ch/people/scientific-assistance/helena-laasch.html), [Tomislav Medic](https://gseg.igp.ethz.ch/people/scientific-assistance/tomislav-medic.html), [Andreas Wieser](https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html)**

Terrestrial laser scanners (TLS) commonly record intensity of the backscattered signal as an auxiliary measurement, which can be related to material properties and used in various applications, such as point cloud segmentation. However, retrieving the material-related information from the TLS intensities is not trivial, as this information is overlayed by other systematic influences affecting the backscattered signal. One of the major factors that needs to be accounted for is the measurement configuration, which is defined by the instrument-to-target distance and angle of incidence (AOI). By obtaining measurement-configuration independent intensity ($I_{\rm MCI}$) material probing, classification, segmentation, and similar tasks can be enhanced. Current methods for obtaining such corrected intensities require additional dedicated measurement set-ups (often in a lab and with specialized targets) and manual work to estimate the effects of distance and AOI on the recorded values. Moreover, they are optimized only for specific datasets comprising a small number of targets with different material properties. This paper presents an automated method for in-situ estimation of $I_{\rm MCI}$, eliminating the need for additional dedicated measurements or manual work. Instead, the proposed method uses overlapping point clouds from different scan stations of an arbitrary scene that are anyway collected during a scanning project. We demonstrate the generalizability of the proposed method across different scenes and instruments, show how the retrieved $I_{\rm MCI}$ values can improve segmentation, and how they increase the comparability of the intensities between different instruments.

Developed by [GSEG, ETH Zurich](https://gseg.igp.ethz.ch/).

## How to install

Create a new environment
```sh
conda create --name <env name> python=3.10
conda activate <env name>
```

```sh
git clone git@github.com:gseg-ethz/InsituRadi.git
cd InsituRadi
pip install .
```

Next, run the main script with a given project folder. The point clouds that should be radiometrically compensated should be in a subfolder called *input_files*.

```sh
python -m InsituRadi -p [project path]
```
