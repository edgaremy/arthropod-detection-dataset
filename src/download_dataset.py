import os
import wget
from tqdm.auto import tqdm
import asyncio
import zipfile

def create_dataset_yaml(dataset_path, dataset_name="ArthroNat"):
    """
    Create a dataset YAML file for YOLO training.
    
    Args:
        dataset_path (str): Absolute path to the dataset root folder
        dataset_name (str): Name of the dataset (used for the YAML filename)
    """
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val
test: images/test

# number of classes
nc: 1

# Classes
names:
  0: "Arthropod"
"""
    
    yaml_file = os.path.join(dataset_path, f"{dataset_name}.yaml")
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"- Created dataset YAML: {yaml_file}")

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
				if os.path.exists(target_dest): # skip already downloaded images
					pbar.update(1)
					continue
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

# Create dataset YAML file:
print("- Creating dataset YAML configuration")
abs_dataset_path = os.path.abspath(dest_file)
create_dataset_yaml(abs_dataset_path)

# Download images from iNaturalist:
print("- Downloading images from iNaturalist (sorry not async, so it will take some time)")
get_images_from_inat(src_csv, dest_file, img_size)
