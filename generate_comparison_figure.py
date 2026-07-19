import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
import matplotlib

# Set global font to Times New Roman for IEEE publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
plt.rcParams['font.size'] = 12

def generate_base_terrain(shape):
    """Generate a procedurally generated geological-looking terrain."""
    np.random.seed(42)  # Use fixed seed for reproducible 'geology'
    noise1 = gaussian_filter(np.random.randn(*shape), sigma=12)
    noise2 = gaussian_filter(np.random.randn(*shape), sigma=4)
    noise3 = gaussian_filter(np.random.randn(*shape), sigma=1.5)
    
    # Create structures and boundaries
    terrain = noise1 * 4.0 + noise2 * 1.5 + noise3 * 0.4
    
    # Add a pseudo-mineral boundary (a sharp edge)
    mask = (np.random.randn(*shape) * 0.1 + np.linspace(0, 1, shape[1])) > 0.6
    terrain[mask] += 2.0
    
    # Normalize to [0.1, 0.8] range
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain = terrain * 0.6 + 0.15
    return terrain

def get_simulated_methods(base):
    """Simulate the denoising outputs based on desired visual characteristics."""
    np.random.seed(42)  # Reproducible noise
    shape = base.shape
    
    # Original: strong vertical detector stripes + random noise
    stripes_orig = np.random.randn(shape[1]) * 0.15
    noise_orig = np.random.randn(*shape) * 0.08
    img_orig = np.clip(base + stripes_orig + noise_orig, 0, 1)
    
    # Physics-Based: large artifacts removed, slight residual striping
    stripes_phys = np.random.randn(shape[1]) * 0.04
    noise_phys = np.random.randn(*shape) * 0.05
    img_phys = np.clip(base + stripes_phys + noise_phys, 0, 1)
    
    # HSID-CNN: improved stripe suppression, slight over-smoothing of details
    img_hsid = gaussian_filter(img_orig, sigma=1.2)
    
    # Proposed Hybrid: effective stripe removal, preserved boundaries/textures, minimal smoothing
    noise_prop = np.random.randn(*shape) * 0.015
    img_prop = np.clip(base + noise_prop, 0, 1)
    
    # N2N4M: reduced random noise, stronger smoothing, lower edge preservation
    img_n2n4m = gaussian_filter(base + np.random.randn(*shape)*0.04, sigma=1.0)
    
    return [img_orig, img_phys, img_hsid, img_prop, img_n2n4m]


# Create mock datasets
shape = (150, 150)
base_L = generate_base_terrain(shape)
np.random.seed(99) # Different terrain for S
base_S = generate_base_terrain(shape)

imgs_L = get_simulated_methods(base_L)
imgs_S = get_simulated_methods(base_S)

# N2N4M is not supported for S sensor
imgs_S[4] = None 

methods = ["Original Observation", "Physics-Based Correction", "HSID-CNN", "Proposed Hybrid Framework", "N2N4M"]
sensors = ["CRISM L Sensor", "CRISM S Sensor"]
all_imgs = [imgs_L, imgs_S]

# Figure setup
fig, axes = plt.subplots(2, 5, figsize=(15, 6), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
fig.suptitle("Qualitative Comparison of CRISM Hyperspectral Image Denoising Methods", fontsize=16, fontweight='bold', y=0.98)

# Intensity limits for normalization (ensure consistent mapping across all panels)
vmin, vmax = 0.0, 1.0

# ROI parameters (red rectangle)
roi_x, roi_y, roi_w, roi_h = 75, 40, 35, 45

for row in range(2):
    for col in range(5):
        ax = axes[row, col]
        img = all_imgs[row][col]
        
        if img is None:
            # Not Supported panel
            ax.imshow(np.zeros(shape), cmap='gray', vmin=0, vmax=1)
            ax.text(shape[1]/2, shape[0]/2, "Not Supported", color='white', ha='center', va='center', fontsize=12)
        else:
            # Display image
            im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', interpolation='none')
            
            # Draw ROI
            rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # Column titles
        if row == 0:
            ax.set_title(methods[col], fontsize=12, pad=10)
            
        # Row labels
        if col == 0:
            ax.set_ylabel(sensors[row], fontsize=12, labelpad=10, fontweight='bold')

# Add single shared colorbar on the right
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Reflectance (I/F)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.subplots_adjust(left=0.05, right=0.9, top=0.88, bottom=0.05)

# Save at high publication quality
plt.savefig("crism_denoising_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig("crism_denoising_comparison.png", dpi=300, bbox_inches='tight')

print("Qualitative comparison figure generated.")
