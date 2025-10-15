import json
import os
import random
import numpy as np
import scipy.stats as stats
import glob

dataset = 'bdd100k'
output_dataset = 'BDD_domain_small'
dataset_dir = f'data'

repeats = [1]
sigmas = [0.1]
seeds = [1, 2, 3]

# Domain order for incremental sampling
dataset_dirs = ['bdd100k_cloudy', 'bdd100k_rainy', 'bdd100k_dawndusk', 'bdd100k_night']
domain_dict = {i: domain for i, domain in enumerate(dataset_dirs)}

# Initialize domain data list
domain_datalist = {}
for i, domain in enumerate(dataset_dirs):
    domain_datalist[i] = []

# Collect samples for each domain
for domain_id, domain_name in enumerate(dataset_dirs):
    domain_path = os.path.join(dataset_dir, domain_name)
    
    # Get all image files for this domain
    images_dir = os.path.join(domain_path, 'images')
    # breakpoint()
    if os.path.exists(images_dir):
        # Find all image files in train* and val* subdirectories
        image_files = (glob.glob(os.path.join(images_dir, "train*/*.jpg")) + 
                      glob.glob(os.path.join(images_dir, "train*/*.png")))
        
        for image_file in image_files:
            split_name = image_file.split('/')[-2]
            base_name = os.path.splitext(image_file.split('/')[-1])[0]  # Remove extension
            
            # Store the relative path from the domain folder
            sample_path = f"{domain_name}/{split_name}/{base_name}"
            domain_datalist[domain_id].append(sample_path)

print(f"Domain sample counts:")
for domain_id, domain_name in enumerate(dataset_dirs):
    print(f"{domain_name}: {len(domain_datalist[domain_id])} samples")

domain_datalist[3] = random.sample(domain_datalist[3], len(domain_datalist[3])//2)
print(len(domain_datalist[3]))
# Generate streams for different configurations
for repeat in repeats:
    for sigma in sigmas:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            
            n_domains = len(dataset_dirs)
            domain_increment_time = np.zeros(n_domains)
            samples_list = []
            
            # Prepare samples for each domain
            for domain_id in range(n_domains):
                datalist = domain_datalist[domain_id].copy()
                random.shuffle(datalist)
                samples_list.append(datalist[:3000])
            
            # Create stream with time-based sampling
            stream = []
            for i in range(n_domains):
                if len(samples_list[i]) > 0:  # Only process domains with samples
                    # Generate times with normal distribution centered at i/n_domains
                    times = np.random.normal(i/n_domains, sigma, size=len(samples_list[i]))
                    choice = np.random.choice(repeat, size=len(samples_list[i]))
                    times += choice
                    
                    for ii, sample in enumerate(samples_list[i]):
                        if choice[ii] >= domain_increment_time[i]:
                            stream.append({
                                'file_name': 'train/' + samples_list[i][ii].split('/')[-1], 
                                'domain': dataset_dirs[i], 
                                'time': times[ii]
                            })
            
            # Sort stream by time
            random.shuffle(stream)
            stream = sorted(stream, key=lambda d: d['time'])
            
            # Prepare final data structure
            data = {
                'domain_dict': domain_dict, 
                'stream': stream, 
                'domain_addition': list(domain_increment_time)
            }
            
            # Create output directory if it doesn't exist
            output_dir = f'collections/{output_dataset}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to JSON file
            output_file = f'{output_dir}/{output_dataset}_sigma{int(sigma*100)}_repeat{repeat}_seed{seed}.json'
            with open(output_file, 'w') as fp:
                json.dump(data, fp, indent=2)
            
            print(f"Generated stream with {len(stream)} samples: {output_file}")