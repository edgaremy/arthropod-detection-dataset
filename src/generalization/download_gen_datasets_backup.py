import os
import wget
from tqdm.auto import tqdm
import asyncio
import zipfile

def create_dataset_yaml(dataset_path, dataset_name):
    """
    Create a dataset YAML file for YOLO training.
    
    Args:
        dataset_path (str): Absolute path to the dataset root folder
        dataset_name (str): Name of the dataset (used for the YAML filename)
    """
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val
#test: images/test

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
	import requests
	os.makedirs(dest_file, exist_ok=True)

	image_dir = os.path.join(dest_file, 'images')
	if not os.path.exists(image_dir):
		os.makedirs(image_dir, exist_ok=True)

	# Load CSV of selected pictures : #taxon_id	#photo_id #extension #observation_uuid
	with open(src_csv, newline='') as csvfile:
		lines = csvfile.read().split("\n")
		# Count valid lines for progress bar
		valid_lines = [l for l in lines[1:] if len(l.split(',')) > 4]
		pbar = tqdm(total=len(valid_lines), desc=f"Downloading {os.path.basename(dest_file)}")
		
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
				
				try:
					response = requests.get(image_url, timeout=30)
					if response.status_code == 200:
						with open(target_dest, 'wb') as f:
							f.write(response.content)
				except Exception as e:
					print(f"Failed to download {image_url}: {e}")
				
				pbar.update(1)
		
		pbar.close()


img_size = "original"  # "small" (240px)/ "medium" (500px)/ "large" (1024px)/ "original" (2024px)

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
		
		# Collect all download tasks for this dataset
		tasks = []
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
				tasks.append(get_image(image_url, target_dest, pbar))
		
		# Wait for all downloads in this dataset to complete before returning
		if tasks:
			try:
				loop = asyncio.get_running_loop()
			except RuntimeError:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
			
			loop.run_until_complete(asyncio.gather(*tasks))
		
		pbar.close()


img_size = "original"  # "small" (240px)/ "medium" (500px)/ "large" (1024px)/ "original" (2024px)

# create three sub-datasets under datasets(generalization) and process each
base_dest = "datasets(generalization)"

dataset_names = ["same_genus", "other_genus", "other_families"]

# First loop: Create directories, unzip labels, and create YAML files
for name in dataset_names:
    print(f"Processing dataset: {name}")
    dest_subdir = os.path.join(base_dest, name)
    os.makedirs(dest_subdir, exist_ok=True)

    # Unzip label file for this sub-dataset
    labels_file = os.path.join("src", "generalization", f"{name}_labels.zip")
    if os.path.exists(labels_file):
        print(f"- Unzipping labels for {name}")
        with zipfile.ZipFile(labels_file, "r") as zip_ref:
            zip_ref.extractall(dest_subdir)
    else:
        print(f"- Warning: labels file not found: {labels_file}")
    
    # Create dataset YAML file
    abs_dataset_path = os.path.abspath(dest_subdir)
    create_dataset_yaml(abs_dataset_path, name)

# Second loop: Download images (done last)
print("\nStarting image downloads...")
for name in dataset_names:
    print(f"Downloading images for dataset: {name}")
    dest_subdir = os.path.join(base_dest, name)
    
    # Download images for this sub-dataset
    src_csv = os.path.join("src", "generalization", f"{name}_images.csv")
    if os.path.exists(src_csv):
        print(f"- Downloading images for {name} from iNaturalist (csv: {src_csv})")
        # get_images_from_inat will create an 'images' subfolder inside dest_subdir
        get_images_from_inat(src_csv, dest_subdir, img_size)
    else:
        print(f"- Warning: images CSV not found: {src_csv}")
