import numpy as np

# all_features = np.zeros((100, 256), dtype="float32")
# with torch.inference_mode():
#     for i, x in enumerate(tqdm(dataloader)):
#         all_features[i*args.batch_size:(i+1)*args.batch_size] = x.cpu().numpy()

# features_real.npy
# features_rendered.npy

features_real = np.load("features_real.npy")
features_rendered = np.load("features_rendered.npy")

assert features_real.shape == features_rendered.shape

# Normalize the feature vectors
features_real_norm = features_real / np.linalg.norm(features_real, axis=1, keepdims=True)
features_rendered_norm = features_rendered / np.linalg.norm(features_rendered, axis=1, keepdims=True)

# Image retrieval: for each real image, find the 10 most similar rendered images using cosine similarity
for i in [2, 3, 24, 67]:
    real_feature = features_real_norm[i-1]
    rendered_features = features_rendered_norm
    
    # Calculate cosine similarity
    cosine_similarities = np.dot(rendered_features, real_feature)
    
    # Get the indices of the most similar and most different rendered images
    closest_indices = np.argsort(-cosine_similarities)  # Sort in descending order
    furthest_indices = np.argsort(cosine_similarities)  # Sort in ascending order
    
    print(f"Real image {i}:")
    print(f"- Most similar rendered images: {closest_indices[:3]+1}")
    print(f"  similarities: {cosine_similarities[closest_indices[:3]].round(2)}")
    
    print(f"- Most different rendered images: {furthest_indices[:3]+1}")
    print(f"  similarities: {cosine_similarities[furthest_indices[:3]].round(2)}")
    
    print(f"- Similarity to corresponding rendered image: {str(cosine_similarities[i-1])[:4]}")








# image retrieval: for each real image, find the 10 most similar rendered images

# features_real: 100 x 256
# features_rendered: 100 x 256

# for i in [2,3,24,67]:
#     real_feature = features_real[i-1]
#     rendered_features = features_rendered
#     distances = np.linalg.norm(rendered_features - real_feature, axis=1)
#     closest_indices = np.argsort(distances)
#     print(f"Real image {i}:")
#     print(f"- Most similar rendered images: {closest_indices[:6]+1}")
#     print(f"  distances: {distances[closest_indices[:6]].round(2)}")
    
#     print(f"- Most different rendered images: {closest_indices[-5:]+1}")
#     print(f"  distances: {distances[closest_indices[-5:]].round(2)}")

#     print(f"- Distance to corresponding rendered image: {str(distances[i-1])[:4]}")