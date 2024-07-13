import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Muat data
data = pd.read_csv("../expdata/ba6.asc", sep="\s+", comment="#")

# Normalisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Lakukan PCA
pca = PCA(n_components=2)  # Anda dapat mengubah jumlah komponen sesuai kebutuhan
principal_components = pca.fit_transform(data_scaled)

# Simpan hasil PCA ke file
pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
pca_df.to_csv("pca_result.asc", index=False, sep=" ")
