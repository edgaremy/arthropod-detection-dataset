"""Plot the ArthroNat taxon repartition as a sunburst chart.

The script scans the YOLO dataset splits under `dataset/images/{train,val,test}`
and counts images directly from filenames. Each image is expected to start with
`<taxon_id>_`, which is matched against the hierarchy CSV.
"""

from __future__ import annotations

import colorsys
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
from pypalettes import load_cmap


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_HIERARCHY_CSV = REPO_ROOT / "src" / "arthro_dataset_hierarchy.csv"
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_HTML_OUTPUT = SCRIPT_DIR / "taxon_repartition.html"
DEFAULT_PNG_OUTPUT = SCRIPT_DIR / "taxon_repartition.png"
DEFAULT_HIERARCHY_LIST = ["phylum", "class", "order", "family"]
SHOW_FIGURE = True
FONT_SIZE = 46
COLORMAP_NAME = "sPBIYlGn"
COLORMAP_START_INDEX = 5
SATURATION_FACTOR = 1.4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def has_kaleido() -> bool:
	return importlib.util.find_spec("kaleido") is not None


def get_custom_color_list() -> list[str]:
	"""Load colormap and adjust saturation."""
	cmap = load_cmap(COLORMAP_NAME)
	color_list = [color[:-2] for color in cmap.colors]  # Remove alpha channel
	
	# Rotate color list by start index
	if COLORMAP_START_INDEX > 0:
		color_list = color_list[COLORMAP_START_INDEX:] + color_list[:COLORMAP_START_INDEX]
	
	# Apply saturation boost
	adjusted_colors = []
	for color_hex in color_list:
		# Convert hex to RGB (0-1 range)
		r = int(color_hex[1:3], 16) / 255.0
		g = int(color_hex[3:5], 16) / 255.0
		b = int(color_hex[5:7], 16) / 255.0
		
		# Convert RGB to HLS
		h, l, s = colorsys.rgb_to_hls(r, g, b)
		
		# Boost saturation
		s = min(1.0, s * SATURATION_FACTOR)
		
		# Convert back to RGB
		r, g, b = colorsys.hls_to_rgb(h, l, s)
		
		# Convert back to hex
		color_hex_adjusted = "#{:02x}{:02x}{:02x}".format(
			int(r * 255), int(g * 255), int(b * 255)
		)
		adjusted_colors.append(color_hex_adjusted)
	
	return adjusted_colors


def count_images_per_taxon(dataset_root: Path) -> dict[int, int]:
	counts: dict[int, int] = {}
	for split in ("train", "val", "test"):
		images_dir = dataset_root / "images" / split
		if not images_dir.is_dir():
			continue

		for image_path in images_dir.iterdir():
			if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
				continue

			taxon_part = image_path.name.split("_", 1)[0]
			try:
				taxon_id = int(taxon_part)
			except ValueError:
				continue

			counts[taxon_id] = counts.get(taxon_id, 0) + 1

	return counts


def build_plottable_dataframe(hierarchy_csv: Path, dataset_root: Path) -> pd.DataFrame:
	df = pd.read_csv(hierarchy_csv)
	df["taxon_id"] = pd.to_numeric(df["taxon_id"], errors="coerce")
	df = df.dropna(subset=["taxon_id"]).copy()
	df["taxon_id"] = df["taxon_id"].astype(int)

	counts = count_images_per_taxon(dataset_root)
	df["count"] = df["taxon_id"].map(counts).fillna(0).astype(int)
	df.fillna("unknown", inplace=True)
	
	# Add Arthropoda as the root phylum for all taxa
	df["phylum"] = "Arthropoda"
	
	return df


def plot_taxon_sunburst(
	df: pd.DataFrame,
	hierarchy_list: list[str],
	output_html: Path | None = None,
	output_png: Path | None = None,
	show: bool = True,
	font_size: int = 32,
) -> None:
	custom_colors = get_custom_color_list()
	
	fig = px.sunburst(
		df,
		path=hierarchy_list,
		values="count",
		color="order",
		template="presentation",
		color_discrete_sequence=custom_colors,
	)

	fig.update_layout(
		font=dict(size=font_size),
		hoverlabel=dict(font_size=24, font_family="DejaVu Sans"),
		transition=dict(duration=0.4, easing="cubic-in-out"),
	)
	fig.update_traces(
		hovertemplate=(
			"<b>%{label}</b><br>"
			"Images: %{value}<br>"
			"Taxon: %{id}<extra></extra>"
		),
		marker=dict(line=dict(color="rgba(20,20,20,0.35)", width=1.2)),
	)

	if show:
		fig.show()

	if output_html is not None:
		output_html.parent.mkdir(parents=True, exist_ok=True)
		fig.write_html(str(output_html))

	if output_png is not None:
		output_png.parent.mkdir(parents=True, exist_ok=True)
		fig.update_layout(paper_bgcolor="rgba(0, 0, 0, 0)", margin=dict(l=0, r=0, t=0, b=0))
		if has_kaleido():
			fig.write_image(str(output_png), width=1200, height=1200, scale=3)
		else:
			print("Warning: kaleido is not installed, skipping PNG export.")


def main() -> None:
	df = build_plottable_dataframe(DEFAULT_HIERARCHY_CSV, DEFAULT_DATASET_ROOT)
	plot_taxon_sunburst(
		df,
		hierarchy_list=DEFAULT_HIERARCHY_LIST,
		output_html=DEFAULT_HTML_OUTPUT,
		output_png=DEFAULT_PNG_OUTPUT,
		show=SHOW_FIGURE,
		font_size=FONT_SIZE,
	)


if __name__ == "__main__":
	main()