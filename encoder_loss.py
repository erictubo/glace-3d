import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def magnitude_loss(features, target_value=1.0, margin=0.15):

    magnitude = torch.mean(torch.norm(features, p=2, dim=1))

    return F.relu(torch.abs(target_value - magnitude) - margin)

    # feature_norms = torch.norm(features, dim=1)
    # losses = F.relu(torch.abs(target_value - feature_norms) - margin)

    # return losses.mean()


def cosine_loss( features_1, features_2, target_value=1, margin=0.1):

    cos_sim = F.cosine_similarity(features_1, features_2, dim=1)

    losses = F.relu(torch.abs(target_value - cos_sim) - margin)

    return losses.mean()


def mse_loss(features_1, features_2, target_value=0.0, margin=0.0, p=1):

    features_1 = F.normalize(features_1, p=p, dim=1)
    features_2 = F.normalize(features_2, p=p, dim=1)

    mse = F.mse_loss(features_1, features_2, reduction='none')

    losses = F.relu(torch.abs(target_value - mse) - margin)

    return losses.mean()


def mae_loss(features_1, features_2, target_value=0.0, margin=0.0, smooth=True, p=1):

    features_1 = F.normalize(features_1, p=p, dim=1)
    features_2 = F.normalize(features_2, p=p, dim=1)

    if smooth:
        mae = F.smooth_l1_loss(features_1, features_2, reduction='none')
    else:
        mae = F.l1_loss(features_1, features_2, reduction='none')

    losses = F.relu(torch.abs(target_value - mae) - margin)

    return losses.mean()


def diversity_loss(features, feature_mask):
    B, C, H, W = features.shape
    
    # Reshape features
    features_reshaped = features.view(B, C, -1)  # Shape: (B, C, H*W)
    
    # Reshape mask_1
    mask_reshaped = feature_mask.view(B, 1, -1)  # Shape: (B, 1, H*W)
    
    # Apply mask_1
    features_masked = features_reshaped * mask_reshaped  # Broadcasting the mask_1
    
    # Normalize features
    features_norm = F.normalize(features_masked, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))
    
    # Compute diversity loss
    eye = torch.eye(C, device=features.device).unsqueeze(0).expand(B, -1, -1)
    diversity_loss = torch.mean((similarity_matrix - eye) ** 2)

    return diversity_loss


def spatial_consistency_loss(features, feature_mask):

    assert len(features.shape) == 4, features.shape
    B, C, H, W = features.shape

    mask_B1HW = feature_mask
    mask_BCHW = mask_B1HW.expand(-1, C, -1, -1)

    assert mask_BCHW.shape == (B, C, H, W), mask_BCHW.shape
    
    # Compute gradients in x and y directions
    grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]
    grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]

    # Update mask_1 to exclude border pixels (where gradients are not valid)
    mask_x = mask_BCHW[:, :, :, 1:] & mask_BCHW[:, :, :, :-1]
    mask_y = mask_BCHW[:, :, 1:, :] & mask_BCHW[:, :, :-1, :]

    # mask_1 gradients
    grad_x_masked = grad_x[mask_x]
    grad_y_masked = grad_y[mask_y]

    # Compute total variation
    return torch.mean(torch.abs(grad_x_masked)) + torch.mean(torch.abs(grad_y_masked))


def triplet_loss(anchor, positive, negative, margin=0.2):

    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2) # positive, negative

    losses = F.relu(distance_positive - distance_negative + margin)

    return losses.mean()


def coords_loss(gt_coords, pred_coords, median=False, visualize=False):
    """
    Compute the loss between predicted and ground truth coordinates.
    """
    assert pred_coords.shape == gt_coords.shape, f"{pred_coords.shape} != {gt_coords.shape}"

    distance = torch.norm(gt_coords - pred_coords, p=2, dim=1)
    valid_coords = (gt_coords.sum(dim=1) != 0)
    distance_valid = distance[valid_coords]

    if visualize:
        difference = distance.cpu().numpy()
        valid = valid_coords.cpu().numpy()
        masked_difference = np.ma.masked_array(difference, mask=~valid)

        cmap = plt.get_cmap('Spectral').reversed()
        cmap.set_bad(color='white')
        B = gt_coords.size(0)
        fig, ax = plt.subplots(B, 3)
        for i in range(B):
            ax[i, 0].imshow(coords_to_colors(gt_coords[i].cpu()))
            ax[i, 1].imshow(coords_to_colors(pred_coords[i].cpu()))
            ax[i, 2].imshow(masked_difference[i], cmap=cmap)
            # add colorbar to difference plot
            cbar = plt.colorbar(ax[i, 2].imshow(masked_difference[i], cmap=cmap, vmin=0), ax=ax[i, 2])
            ax[i, 2].set_title(f"Median: {np.median(masked_difference[i]):.2f}, Mean: {np.mean(masked_difference[i]):.2f}")
        plt.show()

    V = distance_valid.size(0)

    if median:
        loss = distance_valid.median()
    else:
        loss = distance_valid.mean()

    return loss, V


def mask_features(features_list, feature_mask):
    """
    Mask features to valid values only, reshaping to 2D tensor.\\
    Input: features_list (shape BxCxHxW each), feature_mask (shape Bx1xHxW)\\
    Output: valid_features_list (shape MxC each, where M is the number of valid patches, M <= N = B*H*W)
    """

    B, C, H, W = features_list[0].shape

    for features in features_list:
        assert features.shape == (B, C, H, W), features.shape

    def normalize_shape(tensor_in):
        """Bring tensor from shape BxCxHxW to NxC"""
        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)
    
    feature_mask_N1 = normalize_shape(feature_mask)

    assert feature_mask_N1.shape == (B*H*W, 1), feature_mask.shape

    feature_mask_NC = feature_mask_N1.expand(B*H*W, C)

    assert feature_mask_NC.shape == (B*H*W, C), feature_mask_NC.shape

    features_NC_list = [normalize_shape(features) for features in features_list]

    def apply_mask(features_NC, mask_NC):
        valid_features = features_NC[mask_NC]

        N, C = features_NC.shape

        return valid_features.reshape(-1, C)

    valid_features_list = [apply_mask(features_NC, feature_mask_NC) for features_NC in features_NC_list]

    M, C = valid_features_list[0].shape

    for valid_features in valid_features_list:
        assert valid_features.shape == (M, C), valid_features.shape
    
    N = B*H*W
    assert M <= N, f"Masked size {M} larger than {N} = {B}*{H}*{W}"
    if M == N:
        print('M=N so mean feature mask_1 should be equal to 1.0:')
        try: print(feature_mask.float().mean())
        except: print('error in mean calculation')
    
    assert(len(valid_features_list) == len(features_list))

    return valid_features_list, M


def coords_to_colors(coords):
    # 1. Convert coords to numpy array
    coords = coords.permute(1, 2, 0).numpy()

    # 2. Mask out zero values in coords
    mask = np.all(coords == [0., 0., 0.], axis=-1)
    masked_coords = np.ma.masked_array(coords, mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2))

    # 3. Normalize coords to [0, 1]
    min_coords = np.floor(masked_coords.min(axis=(0, 1)))
    max_coords = np.ceil(masked_coords.max(axis=(0, 1)))
    normalized_coords = (masked_coords - min_coords) / (max_coords - min_coords)

    normalized_coords = np.where(mask[:, :, np.newaxis], 1, normalized_coords)

    return normalized_coords.astype(np.float32)