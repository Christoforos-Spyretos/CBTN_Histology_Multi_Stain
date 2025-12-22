# %%
import openslide
from PIL import Image
import matplotlib.pyplot as plt

HE_example = '/local/data2/chrsp39/CBTN_v2/HE/WSI/C17466___7316-38___HandE.svs'
KI67_example = '/local/data2/chrsp39/CBTN_v2/KI67/WSI/C17466___7316-38___KI-67.svs'

HE_slide = openslide.OpenSlide(HE_example)
KI67_slide = openslide.OpenSlide(KI67_example)

HE_region = HE_slide.read_region((0, 0), HE_slide.level_count - 1, HE_slide.level_dimensions[-1])
KI67_region = KI67_slide.read_region((0, 0), KI67_slide.level_count - 1, KI67_slide.level_dimensions[-1])

HE_region = HE_region.convert("RGB")
KI67_region = KI67_region.convert("RGB")

def plot_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

plot_image(HE_region)
plot_image(KI67_region)

# %%
