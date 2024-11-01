import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import plotly.express as px

# Função para verificar se uma cor é próxima de cinza ou branco
def is_gray_or_white(color, threshold=30):
    r, g, b = color
    if abs(r - 255) < threshold and abs(g - 255) < threshold and abs(b - 255) < threshold:
        return True
    if abs(r - g) < threshold and abs(g - b) < threshold and abs(r - b) < threshold:
        return True
    return False

# Função para processar a imagem e calcular as cores predominantes
def process_image(image):
    image = image.convert('RGB')
    image = image.resize((image.width // 4, image.height // 4))
    colors = np.array(image.getdata())
    filtered_colors = np.array([color for color in colors if not is_gray_or_white(color)])

    n_colors = 14
    kmeans = KMeans(n_clusters=n_colors, random_state=0,
                    n_init=10, max_iter=300)
    kmeans.fit(filtered_colors)

    quantized_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    color_count = Counter(labels)
    total_pixels = sum(color_count.values())

    color_df = pd.DataFrame({
        'Color': [tuple(color) for color in quantized_colors],
        'Count': [color_count[i] for i in range(n_colors)]
    })

    color_df['Percentage'] = (color_df['Count'] / total_pixels) * 100

    return image, color_df

# Interface do Streamlit
st.title("Análise de Cores em Imagens")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_processed, results_df = process_image(image)

    # Desconsiderar cores com porcentagem menor que 1%
    results_df = results_df[results_df['Percentage'] >= 1]

    # Verificar se o DataFrame não está vazio
    if not results_df.empty:
        color_map = {str(tuple(color)): f'rgb{tuple(color)}' for color in results_df['Color']}

        fig = px.bar(
            results_df,
            x='Percentage',
            y=results_df['Color'].apply(str),
            orientation='h',
            title='Cores na Imagem por Percentagem',
            labels={'Percentage': 'Percentagem(%)', 'y': 'Cor'},
            text=results_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
            color=results_df['Color'].apply(str),
            color_discrete_map=color_map,
            height=800,
            width=1000,
        )

        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF', font=dict(color='black'))

        st.image(image, caption='Imagem Carregada', use_column_width=True)
        st.plotly_chart(fig)
        st.dataframe(results_df.round(2))
    else:
        st.write("Nenhuma cor significativa encontrada na imagem.")
