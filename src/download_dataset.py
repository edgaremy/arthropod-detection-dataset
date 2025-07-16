import os
import wget
from tqdm.auto import tqdm
import asyncio
import zipfile

"""
This function downloads images from the iNaturalist dataset based on a CSV file containing image metadata.
The CSV file should contain columns for taxon_id, photo_id, extension, observation_uuid and split.
The images will be saved in a specified directory structure based on the split (train, val, test).
"""
def get_images_from_inat(src_csv, dest_file, img_size="original"):
	os.makedirs(dest_file, exist_ok=True)

	def background(f):
		def wrapped(*args, **kwargs):
			try:
				loop = asyncio.get_running_loop()
			except RuntimeError:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
			return loop.run_in_executor(None, f, *args, **kwargs)
			
		return wrapped
		
	@background
	def get_image(image_url, target_dest, pbar):
		wget.download(image_url, target_dest, bar=None)
		pbar.update(1)
		return

	image_dir = os.path.join(dest_file, 'images')
	if not os.path.exists(image_dir):
		os.makedirs(image_dir, exist_ok=True)

	# Load CSV of selected pictures : #taxon_id	#photo_id #extension #observation_uuid
	with open(src_csv, newline='') as csvfile:
		lines = csvfile.read().split("\n")
		pbar = tqdm(total=len(lines), desc="(ASYNC) INAT SCRAPPING")
		for i,row in enumerate(lines):
			data = row.split(',')
			if i > 0 and len(data) > 4:
				taxon_id = data[0]
				photo_id = data[1]
				extension = data[2]
				split = data[4]
				
				# Ensure the split directory exists
				split_dir = os.path.join(image_dir, split)
				os.makedirs(split_dir, exist_ok=True)
				
				target_dest = os.path.join(split_dir, f"{taxon_id}_{photo_id}.{extension}")

				image_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{img_size}.{extension}"
				get_image(image_url, target_dest, pbar)



dest_file = "dataset"
src_csv = "src/dataset_images.csv"
img_size = "original"  # "small" (240px)/ "medium" (500px)/ "large" (1024px)/ "original" (2024px)

print("Starting dataset installation...")

# Unzip label file:
print("- Unzipping labels")
labels_file = "src/dataset_labels.zip"
with zipfile.ZipFile(labels_file, 'r') as zip_ref:
	zip_ref.extractall(dest_file)

# Download images from iNaturalist:
print("- Downloading images from iNaturalist")
get_images_from_inat(src_csv, dest_file, img_size)
