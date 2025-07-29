# Validation of the YOLO model

Once you have [set up Python](../README.md#set-up-python) and [downloaded the initial dataset](../README.md#download-the-dataset), you can run various validation task in order to reproduce our results.

This is done in to separate steps, in order to make plotting scripts faster to run and experiment with:
- First measure every metric needed by using the model on a given validation set, and store everythin in a csv file (done in the [metrics](metrics) folder). This part takes the longest to run.
- Second, use the outputed csv file to display metrics in interpretable plots.

## 1. Generate metric csv

Note: This part can be skipped if you just want to play around with the plots using the already generated csv files.

*Still a Work in Progress*


## 2. Plot metrics using csv

### Hierarchical Performance

```bash
python validation/plot_from_metrics/hierarchical_perfs/plot_hierarchical_perfs.py
```
 Running this script allows you to get the plots in the [`hierarchical_perfs/plots`](plot_from_metrics/hierarchical_perfs/plots) folder, such as:

 <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/hierarchical_perfs/plots/class_perfs_yolo11l.png?raw=true" width="400" align="center">


### Performance according to image properties


 Running the scripts below allows you to get the plots in the [`perfs_vs_img_properties/plots`](plot_from_metrics/perfs_vs_img_properties/plots) folder, such as:

```bash
python validation/plot_from_metrics/perfs_vs_img_properties/plot_bbox_size_perf.py
```

 <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_size_IoU_11l.png?raw=true" width="300" align="center">

or

```bash
python validation/plot_from_metrics/perfs_vs_img_properties/plot_bbox_number_perf.py
```
 <img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/validation/plot_from_metrics/perfs_vs_img_properties/plots/bbox_number_F1_11l.png?raw=true" width="300" align="center">

## Download additional validation datasets

TODO

#
# 🐞 🐜 🦋 🦗 🐝 🕷️ 🐛 🪰 🪲