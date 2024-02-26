import numpy as np

# np.savez_compressed('character_font.npz')
data = np.load('character_font.npz')
print(data['images'].shape)