
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Especifica la ruta del archivo CSV en tu directorio local
data_path = "1Dry_Bean_Dataset.csv"  # Cambia esta ruta a tu ubicación real

# Renombrar las columnas basadas en sus características
columns = ['Area	Perimeter',	'MajorAxisLength',	'MinorAxisLength',	'AspectRation',	'Eccentricity',	'ConvexArea',	'EquivDiameter',	'Extent	Solidity',	'roundness',	'Compactness	ShapeFactor1',	'ShapeFactor2',	'ShapeFactor3',	'ShapeFactor4',	'Class']


# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(data_path, names=columns, header=0)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Convertir la columna 'Class' en variable categórica numérica
df['Class'] = df['Class'].astype('category').cat.codes

# Usar StandardScaler para normalizar los datos (excluyendo la columna 'Class')
X = df.drop('Class', axis=1).values
X = StandardScaler().fit_transform(X)

# Definir colores personalizados para cada clase
colors = {0: 'red', 1: 'BlueViolet', 2: 'Turquoise', 3:'yellow', 4:'DarkGray', 5:'Magenta', 6:'Coral'}  # Cambia los colores según tus preferencias
labels = {0: 'SEKER', 1: 'BARBUNYA', 2: 'BOMBAY', 3:'CALI', 4:'HOROZ', 5:'SIRA', 6:'DERMASON',}

# PCA 2D ------------------------------------------------------------------
pca = PCA(n_components=2)
principalComponents1 = pca.fit_transform(X)


PCA_dataset1 = pd.DataFrame(data=principalComponents1, columns=['component1', 'component2'])

# Visualizando los efectos del PCA con diferenciación de color
plt.figure(figsize=(10, 10))
plt.scatter(PCA_dataset1['component1'], PCA_dataset1['component2'], c=df['Class'].map(colors), cmap='viridis', s=5)
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('Principales Componentes en 2D (PCA)')

# Crear leyenda manualmente
handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=10) for i in labels]
plt.legend(handles=handles, title='Tipo de Frijol')

plt.show()


#PCA 3D------------------------------------------------------------------
pca = PCA(n_components=3)
principalComponents2 = pca.fit_transform(X)

# Crear un DataFrame con los tres componentes principales
PCA_dataset2 = pd.DataFrame(data=principalComponents2, columns=['component1', 'component2', 'component3'])

# Visualizar los resultados del PCA 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plt.title('Principales Componentes en 3D (PCA)')

# Agregar color basado en la clase
scatter = ax.scatter(
    PCA_dataset2['component1'], 
    PCA_dataset2['component2'], 
    PCA_dataset2['component3'], 
    c=df['Class'].map(colors), 
    cmap='viridis',
    s=5
)
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')

# Añadir una barra de color para identificar las clases

# Crear leyenda manualmente
handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=10) for i in labels]
plt.legend(handles=handles, title='Tipo de Frijol')
plt.show()



#TSNE 2D-----------------------------------------------------------------------------------

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results1 = tsne.fit_transform(X)

# Creating a dataframe featuring the two principal components that we acquired through t-SNE.
tsne_dataset1 = pd.DataFrame(data = tsne_results1, columns = ['component1', 'component2'] )
tsne_dataset1.head()

# Extracting the two features from above in order to add them to the dataframe.
tsne_component1 = tsne_dataset1['component1']
tsne_component2 = tsne_dataset1['component2']

# Visualizando los efectos del t-SNE con diferenciación de color
plt.figure(figsize=(10, 10))
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('t-SNE 2D')

# Agregar color basado en la clase
plt.scatter(tsne_dataset1['component1'], tsne_dataset1['component2'], c=df['Class'].map(colors), cmap='viridis', s=5)

# Crear leyenda manualmente
handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=10) for i in labels]
plt.legend(handles=handles, title='Tipo de Frijol')
plt.show()



#TSNE 3D------------------------------------------------------------------------------------
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter_without_progress=300)
tsne_results2 = tsne.fit_transform(X)


# Creating a dataframe featuring the three Principal components that we acquired through t-SNE.
tsne_dataset2 = pd.DataFrame(data = tsne_results2, columns = ['component3', 'component4', 'component5'] )
tsne_dataset2.head()

# Extracting the three features from above in order to add them to the dataframe.
tsne_component3 = tsne_dataset2['component3']
tsne_component4 = tsne_dataset2['component4']
tsne_component5 = tsne_dataset2['component5']

# Visualizing the 3D t-SNE.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
plt.title('3D T-distributed Stochastic Neighbor Embedding (TSNE)')

# Agregar color basado en la clase (usando df['class'])
scatter = ax.scatter(
    xs=tsne_component1, 
    ys=tsne_component2, 
    zs=tsne_component3,
    c=df['Class'].map(colors),  # Usar la clase como color
    cmap='viridis',  # Puedes cambiar el mapa de colores si lo deseas
    s=5
)

ax.set_xlabel('tsne-one')
ax.set_ylabel('tsne-two')
ax.set_zlabel('tsne-three')


# Crear leyenda manualmente
handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=10) for i in labels]
plt.legend(handles=handles, title='Tipo de Frijol')
plt.show()

# UMAP  2D--------------------------------------------------------------------------
# Implementing UMAP.
embedding = umap.UMAP(n_neighbors=50,min_dist=0.3,metric='correlation').fit_transform(X)

umap_component1 = embedding[:,0]
umap_component2 = embedding[:,1]

# Visualizando los efectos del UMAP con diferenciación de color
plt.figure(figsize=(10, 10))
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('2D Uniform Manifold Approximation and Projection (UMAP)')

# Agregar color basado en la clase
scatter = plt.scatter(umap_component1, umap_component2, c=df['Class'].map(colors), cmap='viridis', s=5)  # Usar la clase como color

# Crear leyenda manualmente
handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=10) for i in labels]
plt.legend(handles=handles, title='Tipo de Frijol')
plt.show()



# UMAP 3D-----------------------------------------------------------------------------
embedding2 = umap.UMAP(n_components=3,n_neighbors=50,min_dist=0.3,metric='correlation').fit_transform(X)

umap_component3 = embedding2[:,0]
umap_component4 = embedding2[:,1]
umap_component5 = embedding2[:,2]

# Visualizing the effects of the 3D UMAP.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
plt.title('3D Uniform Manifold Approximation and Projection (UMAP)')

# Agregar color basado en la clase
scatter = ax.scatter(
    xs=umap_component3, 
    ys=umap_component4, 
    zs=umap_component5, 
    c=df['Class'].map(colors),  # Usar la clase como color
    cmap='viridis',  # Cambia el mapa de colores si lo deseas
    s=5
)

ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')

# Crear leyenda manualmente
handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=10) for i in labels]
plt.legend(handles=handles, title='Tipo de Frijol')
plt.show()