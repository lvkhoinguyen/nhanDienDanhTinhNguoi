import os
import pickle

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms


class FaceDatabaseCreator:
    def __init__(self, cropped_data_path="../datasets/train_cropped"):
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
    
    def create_database(self, samples_per_person=50):
        """Create face database from trained_cropped folder"""
        
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
            
            # limit samples per person
            if len(person_indices) > samples_per_person:
                person_indices = person_indices[:samples_per_person]
            
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
        os.makedirs('models', exist_ok=True)
        
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
        print("📊 Statistics:")
        print(f"  - Total people: {len(dataset.classes)}")
        print(f"  - Total embeddings: {len(all_embeddings)}")
        print(f"  - Average embeddings per person: {len(all_embeddings) / len(dataset.classes):.1f}")
        print("\n💾 Saved files:")
        print("  - models/face_database_all.pkl (all embeddings)")
        print("  - models/face_database_avg.pkl (average embeddings)")
        print("  - models/face_database_detailed.pkl (detailed data)")
        
        return database, avg_database, detailed_database

def main():
    creator = FaceDatabaseCreator("../datasets/train_cropped")
    creator.create_database(samples_per_person=100)

if __name__ == "__main__":
    main()
