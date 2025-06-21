import os
import pickle

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms


class FaceDatabaseCreator:
    def __init__(self, cropped_data_path="../datasets/train_augmented"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cropped_data_path = cropped_data_path
        
        #  pre-trained model facenet
        self.resnet = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        ).to(self.device)
        self.resnet.eval()
        
        print(f"Using device: {self.device}")
        print(f"Loading faces from: {cropped_data_path}")
    
    def create_database(self, samples_per_person=None, min_samples=10, max_samples=None, use_all_available=True):
        """Create face database from trained_cropped folder
        
        Args:
            samples_per_person: Fixed number per person (legacy mode)
            min_samples: Minimum samples required per person
            max_samples: Maximum samples to use per person (None = no limit)
            use_all_available: If True, use all available images (respecting max_samples)
        """
        
        # transforms for facenet
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # load dataset
        dataset = datasets.ImageFolder(self.cropped_data_path, transform=transform)
        
        print(f"Found {len(dataset.classes)} people: {dataset.classes}")
        print(f"Total images: {len(dataset)}")
        
        # store for embeddings later
        embeddings_per_person = {}
        all_embeddings = []
        all_names = []
        
        # process each person
        for class_idx, class_name in enumerate(dataset.classes):
            print(f"\nProcessing {class_name}...")
            
            # get all images for this person
            person_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
            total_available = len(person_indices)
            
            # determine how many samples to use
            if samples_per_person is not None:
                # Legacy mode: fixed number per person
                samples_to_use = min(samples_per_person, total_available)
                person_indices = person_indices[:samples_to_use]
                print(f"  Using {samples_to_use}/{total_available} images (fixed mode)")
            elif use_all_available:
                # use all available images (respecting max_samples)
                if max_samples and total_available > max_samples:
                    person_indices = person_indices[:max_samples]
                    samples_to_use = max_samples
                    print(f"  Using {samples_to_use}/{total_available} images (capped at max_samples)")
                else:
                    samples_to_use = total_available
                    print(f"  Using all {samples_to_use} available images")
            else:
                # use min_samples to max_samples range
                samples_to_use = min(max_samples or total_available, total_available)
                if samples_to_use < min_samples:
                    print(f"  ⚠️  Warning: Only {total_available} images available, less than min_samples ({min_samples})")
                person_indices = person_indices[:samples_to_use]
                print(f"  Using {samples_to_use}/{total_available} images")
            
            # skip if not enough samples
            if total_available < min_samples:
                print(f"  ❌ Skipping {class_name}: only {total_available} images (min required: {min_samples})")
                continue
            
            person_embeddings = []
            
            # process each image
            for idx in person_indices:
                img, _ = dataset[idx]
                img = img.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.resnet(img)
                    embedding = embedding.cpu().numpy().flatten()
                    person_embeddings.append(embedding)
                    all_embeddings.append(embedding)
                    all_names.append(class_name)
            
            embeddings_per_person[class_name] = person_embeddings
            print(f"  Created {len(person_embeddings)} embeddings for {class_name}")
        
        # compute average embeddings per person
        average_embeddings = {}
        for person, embeddings in embeddings_per_person.items():
            avg_embedding = np.mean(embeddings, axis=0)
            average_embeddings[person] = avg_embedding
        
        # save databases
        os.makedirs('../models', exist_ok=True)
        
        # save all individual embeddings
        database = {
            'embeddings': all_embeddings,
            'names': all_names,
            'classes': dataset.classes
        }
        with open('../models/face_database_all_v2.pkl', 'wb') as f:
            pickle.dump(database, f)
        
        # save average embeddings (smaller, faster for recognition)
        avg_database = {
            'embeddings': list(average_embeddings.values()),
            'names': list(average_embeddings.keys()),
            'classes': dataset.classes
        }
        with open('../models/face_database_avg_v2.pkl', 'wb') as f:
            pickle.dump(avg_database, f)
        
        # save detailed embeddings per person
        detailed_database = {
            'embeddings_per_person': embeddings_per_person,
            'average_embeddings': average_embeddings,
            'classes': dataset.classes
        }
        with open('../models/face_database_detailed_v2.pkl', 'wb') as f:
            pickle.dump(detailed_database, f)
        
        print("\n✅ Database created successfully!")
        print("📊 Final Statistics:")
        print(f"  - Total people: {len(embeddings_per_person)}")
        print(f"  - Total embeddings: {len(all_embeddings)}")
        
        # show per-person statistics
        print("\n📋 Embeddings per person:")
        for person, embeddings in embeddings_per_person.items():
            count = len(embeddings)
            print(f"   {person:12} : {count:4d} embeddings")
        
        min_emb = min(len(emb) for emb in embeddings_per_person.values()) if embeddings_per_person else 0
        max_emb = max(len(emb) for emb in embeddings_per_person.values()) if embeddings_per_person else 0
        avg_emb = len(all_embeddings) / len(embeddings_per_person) if embeddings_per_person else 0
        
        print(f"\n📈 Range: Min={min_emb}, Max={max_emb}, Avg={avg_emb:.1f} embeddings per person")
        
        print("\n💾 Saved files:")
        print("  - models/face_database_all_v2.pkl (all embeddings)")
        print("  - models/face_database_avg_v2.pkl (average embeddings)")
        print("  - models/face_database_detailed_v2.pkl (detailed data)")
        
        return database, avg_database, detailed_database

    def analyze_dataset(self):
        """Analyze the dataset to see distribution of images per person"""
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(self.cropped_data_path, transform=transform)
        
        print(f"\n📊 Dataset Analysis for: {self.cropped_data_path}")
        print("=" * 60)
        
        person_counts = {}
        for _, label in dataset.samples:
            person_name = dataset.classes[label]
            person_counts[person_name] = person_counts.get(person_name, 0) + 1
        
        # sort by count for better visualization
        sorted_counts = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)
        
        total_images = sum(person_counts.values())
        min_images = min(person_counts.values())
        max_images = max(person_counts.values())
        avg_images = total_images / len(person_counts)
        
        print(f"Total people: {len(dataset.classes)}")
        print(f"Total images: {total_images}")
        print(f"Images per person - Min: {min_images}, Max: {max_images}, Avg: {avg_images:.1f}")
        print("\nPer person breakdown:")
        print("-" * 30)
        
        for person, count in sorted_counts:
            bar = "█" * min(20, count // 10)
            print(f"{person:12} : {count:4d} images {bar}")
        
        print("\n💡 Recommendations:")
        if max_images - min_images > 100:
            print(f"   • Large variation ({min_images}-{max_images}). Consider using 'use_all_available=True'")
            print(f"   • Or set max_samples={int(avg_images * 1.2)} to balance dataset")
        else:
            print(f"   • Fairly balanced dataset. Fixed samples_per_person={min_images} would work well")
        
        return person_counts

def main():
    creator = FaceDatabaseCreator("../datasets/train_augmented")
    
    print("🔍 Analyzing dataset first...")
    creator.analyze_dataset()
    
    print("\n" + "="*60)
    print("🚀 Creating database...")
    
    # Option 1: Use all available images (recommended for best accuracy)
    creator.create_database(use_all_available=True)
    
    # Option 2: Use all available images but cap at maximum
    # creator.create_database(use_all_available=True, max_samples=300)
    
    # Option 3: Use minimum 50, maximum 250 per person
    # creator.create_database(min_samples=50, max_samples=250, use_all_available=False)
    
    # Option 4: Legacy mode (your current approach)
    # creator.create_database(samples_per_person=200)
    
    print("\n" + "="*50)
    print("💡 TIP: Try different modes by uncommenting the options above:")
    print("   - Option 1: Best accuracy (uses all available images)")
    print("   - Option 2: Good balance (uses all but caps at max)")
    print("   - Option 3: Controlled range (between min-max)")
    print("   - Option 4: Fixed number (your previous default)")

if __name__ == "__main__":
    main()
