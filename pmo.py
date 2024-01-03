import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import datetime
from scipy.stats import norm
import scipy.stats as stats
import pulp
from pulp import LpProblem, LpMaximize
#from pulp import *
# Função para realizar a simulação de Monte Carlo
df2 = pd.DataFrame(columns=["Proposta", "Marco"])
def simular_monte_carlo(media_projeto, desvio_projeto, media_atividades, desvio_atividades, n_simulacoes=10000):
    resultados_projeto = np.zeros(n_simulacoes)
    resultados_atividades = np.zeros(n_simulacoes)

    for i in range(n_simulacoes):
        tempo_projeto = np.random.normal(loc=media_projeto, scale=desvio_projeto)
        tempo_atividades = np.random.normal(loc=media_atividades, scale=desvio_atividades)
        resultados_projeto[i] = tempo_projeto
        resultados_atividades[i] = tempo_atividades
    df2 = pd.DataFrame({"Proposta": resultados_atividades, "Marco": resultados_projeto})
    



    prob_projeto_entrega = sum(df2["Marco"] > df2["Proposta"]) / n_simulacoes

    return prob_projeto_entrega, resultados_projeto, resultados_atividades
# Configurar a largura total da página

st.set_page_config(layout="wide")
def calcular_pontuacao(df, pesos):
    X_norm = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    X_pond = X_norm * np.array(pesos)
    pontuacao = X_pond.sum(axis=1)
    return pontuacao

# Função para calcular o desempenho normalizado para um critério específico
def calcular_desempenho_normalizado(df, crit_selected):
    for crit in crit_selected:
        media_j = df[crit].mean()
        desvio_j = df[crit].std()
        df[f'Desempenho Normalizado {crit}'] = df.apply(lambda row: norm.cdf(row[crit], loc=media_j, scale=desvio_j), axis=1)

# Carregue o DataFrame com os dados do arquivo
df = pd.read_excel("p1.xlsx")
df.rename(columns={'Observação': 'Projeto'}, inplace=True)

# Conteúdo do "sigbar" (à esquerda)
st.sidebar.title('Projetos, Processos e Tecnologia')
st.sidebar.image('LM4.png', caption='', use_column_width=True)
paginaSelecionada = st.sidebar.selectbox('Escolha uma opção', ['Análise Multicritério','Método ProPPAGA','Programação Inteira','Simulação de Entrega','Otimização de Processo'])


# Título da aplicação

if paginaSelecionada == 'Análise Multicritério':
    st.title('Aprendendo Métodos de Tomada de Decisão em Pesquisa Operacional ')
    # Layout em duas colunas
    col1, col2 = st.columns(2)

    # Coluna da esquerda com a imagem
    with col1:
        st.image('imagem_transport_method.png', caption='Exemplo de Método de Transporte', use_column_width=True)

    # Coluna da direita com o texto
    with col2:
        st.markdown("""
        <div style='font-size: 24px; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 40vh;'>
        Na indústria, tomar decisões de negócios é essencial. Essas decisões podem incluir escolher a melhor forma de distribuir produtos, 
        otimizar custos de transporte e considerar a sustentabilidade. Uma técnica útil para auxiliar nesse processo é o Método AHP Gaussiano, 
        que ajuda a comparar e priorizar diversos critérios de forma hierárquica. 
        Com ele, é possível tomar decisões mais informadas e eficazes.
        </div>
        """, unsafe_allow_html=True)
    # Título da aplicação
    tamanho_fonte = 24
    st.subheader('Método AHP Gaussiano na Logística Automobilística')

    # Texto de Introdução
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>O Método AHP Gaussiano é uma técnica útil para tomar decisões.</p>", unsafe_allow_html=True)

    # Explicação do Método
    st.header('Como Funciona o Método AHP Gaussiano?')
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>1. Identificamos critérios importantes, como custo de transporte, tempo de entrega e sustentabilidade.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>2. Comparamos esses critérios para entender quais são mais importantes em relação aos outros.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>3. Usando matemática computacional, calculamos a importância de cada critério.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>4. Isso nos ajuda a tomar decisões melhores, escolhendo a melhor maneira de distribuir produtos.</p>", unsafe_allow_html=True)

    # Benefícios
    st.header('Benefícios do Método AHP Gaussiano')
    # Mensagens com tamanho de fonte personalizado
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>- Ajuda a tomar decisões mais informadas.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>- Economiza dinheiro ao otimizar a distribuição.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>- Contribui para práticas mais sustentáveis.</p>", unsafe_allow_html=True)

    # Conclusão
    st.header('Exemplo Prático:')

    # Mensagem principal com tamanho de fonte personalizado
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>Considere a tabela a seguir, que representa os critérios para um grupo de projetos:</p>", unsafe_allow_html=True)

    # Rodapé com tamanho de fonte personalizado
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>Este é um exemplo simples para ilustrar o conceito do Método AHP Gaussiano.</p>", unsafe_allow_html=True)
    df = pd.read_excel("p1.xlsx")
    df.rename(columns={'Observação': 'Projeto'}, inplace=True)
    df1=df
    # Salvar a primeira coluna (caso haja) e excluí-la temporariamente
    primeira_coluna = None
    if df.columns[0] != df._get_numeric_data().columns[0]:
        primeira_coluna = df[df.columns[0]]
        df = df.drop(columns=[df.columns[0]])

    critrios_numericos = df.select_dtypes(include=[float, int]).columns.tolist()

    # Defina os critérios padrão selecionados
    critrios_selecionados_default = ["Custo de transporte (R$)", "Tempo de entrega (dias)", "Risco (nota de 0 a 10)", "Demanda por qualificação profissional (número de vagas/ano)"]

    # Use o multiselect com os critérios padrão selecionados
    # Use o multiselect com tamanho de fonte personalizado
    st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>Primeiro, o decisor escolhe os critérios monotônicos, ou seja, os critérios em que um valor menor é considerado melhor:</p>", unsafe_allow_html=True)
    critrios_selecionados = st.multiselect("", critrios_numericos, default=critrios_selecionados_default)

    # Botão "Modificar Critérios"
    if st.button("Antes de aplicar o modelo AHP clique aqui para modificar os valores da base de dados para os critérios escolhidos "):
        # Modificar os critérios selecionados para 1/valor (somente para critérios numéricos)
        for criterio in critrios_selecionados:
            if df[criterio].dtype in [float, int]:
                df[criterio] = 1 / df[criterio]

        # Restaurar a primeira coluna (caso exista)
        if primeira_coluna is not None:
            df = pd.concat([primeira_coluna, df], axis=1)

        # Exibir o DataFrame modificado após a modificação inicial
        st.write("DataFrame Modificado após Modificação Inicial:")
    df_display = df.copy()  # Fazer uma cópia do DataFrame original
    for coluna in critrios_numericos:
        df_display[coluna] = df_display[coluna].apply(lambda x: f"{x:.10f}")  # Formatar como float com 10 casas decimais

    st.table(df_display)
    # Botão "2ª Etapa" (Soma e Nova Modificação)
    # Botão "2ª Etapa" (Soma e Nova Modificação)
    if st.button("Aplicar Método AHP para visualizar os resultados"):
        # Realizar a segunda etapa de modificação dos valores (normalização) para critérios numéricos
        df_normalizado = df.copy()  # Criar um novo DataFrame normalizado

        for criterio in critrios_numericos:
            soma_coluna = df_normalizado[criterio].sum()
            df_normalizado[criterio] = df_normalizado[criterio] / soma_coluna  # Dividir pelo valor da soma da coluna

        # Exibir o DataFrame após a segunda etapa (normalizado)
        st.write("DataFrame Após a 2ª Etapa (Normalizado):")
        df_display_normalizado = df_normalizado.copy()
        for coluna in critrios_numericos:
            df_display_normalizado[coluna] = df_display_normalizado[coluna].apply(lambda x: f"{x:.10f}")  # Formatar como float com 10 casas decimais
            df_display_normalizado['Projeto'] = primeira_coluna
            df_display_normalizado_copy2 = df_display_normalizado.drop(columns=['Projeto'])

        # Adicione a coluna primeira_coluna no início do DataFrame copiado
        df_display_normalizado_copy2.insert(0, 'Projeto', primeira_coluna)

        st.table(df_display_normalizado_copy2)
        # Calcular média e desvio padrão para cada coluna do DataFrame normalizado
        st.write("Média e Desvio Padrão para Cada Coluna do DataFrame Normalizado:")

        fator_g_list = []

        # Loop para calcular os valores Fator_G_j e armazená-los na lista
        for coluna in critrios_numericos:
            media = np.mean(df_normalizado[coluna])
            desvio = np.std(df_normalizado[coluna])
            Fator_G = desvio / media
            fator_g_list.append(Fator_G)
            #st.write(f"{coluna}:")
            #st.write(f"Média: {media:.5f}")
            #st.write(f"Desvio Padrão: {desvio:.5f}")
            #st.write(f"Fator Gaussiano: {Fator_G:.5f}")

        # Calcular a soma dos valores Fator_G_j
        soma_fator_g = sum(fator_g_list)

        # Calcular e exibir os pesos W_j para cada critério
        for i, coluna in enumerate(critrios_numericos):
            W_j = fator_g_list[i] / soma_fator_g
            #st.write(f"Peso W_{coluna}: {W_j:.5f}")
        resultados = {
                'Critério': critrios_numericos,
                'Média': [np.mean(df_normalizado[coluna]) for coluna in critrios_numericos],
                'Desvio Padrão': [np.std(df_normalizado[coluna]) for coluna in critrios_numericos],
                'Fator Gaussiano (Fator_G)': fator_g_list,
                'Peso (W_j)': [W_j / soma_fator_g for W_j in fator_g_list]
            }

            # Crie um DataFrame com os resultados
        df_resultados = pd.DataFrame(resultados)

            # Exiba a tabela no Streamlit

        st.table(df_resultados)
        # Suponha que você tem um DataFrame chamado df_display_normalizado com os valores normalizados

        # Lista para armazenar os valores de Vi
        # Suponha que df_display_normalizado seja seu DataFrame
        df_display_normalizado = df_display_normalizado.apply(pd.to_numeric, errors='coerce')

        # Depois, você pode prosseguir com o cálculo de Vi
        v_i_list = []

        for index, row in df_display_normalizado.iterrows():
            vi = sum(row[coluna] * W_j for coluna, W_j in zip(critrios_numericos, fator_g_list))
            v_i_list.append(vi)

        df_display_normalizado['Desempenho global'] = v_i_list
        df_display_normalizado['Projeto'] = primeira_coluna
        df_display_normalizado_copy = df_display_normalizado.drop(columns=['Projeto'])

        # Adicione a coluna primeira_coluna no início do DataFrame copiado
        df_display_normalizado_copy.insert(0, 'Projeto', primeira_coluna)



        # Agora você pode acessar os valores de Vi no DataFrame df_display_normalizado


        # Adicione os valores Vi ao DataFrame df_normalizado


        # Plotar as curvas normais
        plt.figure(figsize=(10, 6))
        for coluna in critrios_numericos:
            media = np.mean(df_normalizado[coluna])
            desvio = np.std(df_normalizado[coluna])
            x = np.linspace(media - 3 * desvio, media + 3 * desvio, 100)
            #plt.plot(x, norm.pdf(x, media, desvio), label=coluna)
            plt.plot(x, norm.pdf(x, media, desvio), label=f'{coluna}')

        col10, col20 = st.columns(2)
        plt.title("Curvas Normais para Critérios Normalizados")
        plt.xlabel("Valores")
        plt.ylabel("")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Exibir o gráfico com a legenda fora da área de plotagem

            #
            # Coluna da esquerda com a imagem
        with col10:
            col10.pyplot(plt)

        # Coluna da direita com o texto
        with col20:
            st.markdown("""
            <div style='font-size: 24px; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh;'>
            No gráfico à esquerda, cada curva representa o peso associado a um critério. Quanto maior a cauda da curva, maior é o peso atribuído a esse critério. Essas curvas gaussianas refletem a dispersão dos pesos para diferentes critérios.O método aplicado aqui é o AHP Gaussiano devido à sua capacidade de modelar a distribuição de pesos dos critérios de forma probabilística por meio de curvas gaussianas. Isso permite uma representação mais detalhada e realista da incerteza associada à atribuição de pesos aos critérios, contribuindo para uma tomada de decisão mais precisa e robusta.
            </div>
            """, unsafe_allow_html=True)
        st.title("Resultado Análise Multicritério Método AHP Gaussiano  ")
        df_display_normalizado_copy = df_display_normalizado_copy.sort_values(by='Desempenho global', ascending=False)
        df_display_normalizado_copy['Desempenho %'] = df_display_normalizado_copy['Desempenho global'].apply(lambda x: f'{x:.2%}')
        # Selecionar a primeira, a penúltima e a última coluna
        colunas_selecionadas = df_display_normalizado_copy.columns[[0, -2, -1]]

        # Criar um novo DataFrame com as colunas selecionadas
        df_selecionado = df_display_normalizado_copy[colunas_selecionadas]



        # Exiba o DataFrame no aplicativo Streamlit
        
        col2x,col1x = st.columns(2)

        # Coluna da esquerda com a imagem
        with col1x:
           st.table(df_selecionado)
        # Coluna da direita com o texto
        with col2x:
            st.markdown("""
            <div style='font-size: 24px; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh;'>
           Note que foram geradas duas novas colunas de dados que representam o desempenho de cada projeto considerando todos os critérios avaliados. Essas colunas refletem a avaliação quantitativa dos projetos com base no Método AHP Gaussiano, que leva em conta a importância relativa de cada critério na tomada de decisão. Isso nos fornece uma medida objetiva do desempenho de cada projeto em relação aos outros, ajudando na seleção e priorização de projetos.
            </div>
            """, unsafe_allow_html=True)
        st.title("Gráfico Radar - Desempenho")
        c1,c2=st.columns(2)
        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(6, 6))

        # Definir os dados para o gráfico de radar
        projetos = df_display_normalizado_copy["Projeto"]
        desempenho = df_display_normalizado_copy["Desempenho global"]

        # Calcular o ângulo para cada ponto no gráfico de radar
        num_pontos = len(projetos)
        angulos = [n / float(num_pontos) * 2 * 3.14159265359 for n in range(num_pontos)]
        angulos += angulos[:1]

        # Adicionar o primeiro valor no final para fechar o gráfico
        desempenho = list(desempenho)
        desempenho += desempenho[:1]

        # Plotar o gráfico de radar
        plt.polar(angulos, desempenho, marker='o')

        # Personalizar as labels do eixo x (projetos)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(projetos, fontsize=8, rotation=90)

        # Exibir o gráfico no aplicativo Streamlit
        st.pyplot(fig)

        # Definir as categorias (projetos) e os valores (Desempenho %)
        categorias = df_display_normalizado_copy["Projeto"]
        valores = df_display_normalizado_copy["Desempenho global"]



        # Obter categorias únicas
        categorias_unicas = categorias.unique()

        # Número de categorias únicas
        num_categorias = len(categorias_unicas)

        # Calcular os ângulos para cada categoria única no gráfico de radar
        angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
        angulos += angulos[:1]

        # Mapear valores de desempenho para os ângulos correspondentes
        valores_mapeados = [valores[categorias == categoria].mean() for categoria in categorias_unicas]

        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})

        # Plotar os valores mapeados para cada categoria única
        ax.fill(angulos, valores_mapeados + valores_mapeados[:1], 'b', alpha=0.1)

        # Configurar as labels do eixo x
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias_unicas, fontsize=8, rotation=15)  # Ajuste o tamanho e a rotação aqui

        # Exibir o gráfico no aplicativo Streamlit
        st.pyplot(fig)
        # Título
        st.title("Gráfico Radar -Tempo de entrega & Retorno sobre o investimento (ROI)(R$)")
        c5,c6=st.columns(2)
        st.title("Gráfico Radar - Sustentabilidade & Nível de Automação")
        c7,c8=st.columns(2)
        st.title("Gráfico Radar - Redução de emissões de CO2 (g/km) & Risco")
        c9,c10=st.columns(2)
        # Definir as categorias (projetos) e os valores (Desempenho %)
        categorias = df_display_normalizado_copy["Projeto"]
        valores = df_display_normalizado_copy["Retorno sobre o investimento (ROI)"]



        # Obter categorias únicas
        categorias_unicas = categorias.unique()

        # Número de categorias únicas
        num_categorias = len(categorias_unicas)

        # Calcular os ângulos para cada categoria única no gráfico de radar
        angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
        angulos += angulos[:1]

        # Mapear valores de desempenho para os ângulos correspondentes
        valores_mapeados = [valores[categorias == categoria].mean() for categoria in categorias_unicas]

        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})

        # Plotar os valores mapeados para cada categoria única
        ax.fill(angulos, valores_mapeados + valores_mapeados[:1], 'b',color='red', alpha=0.3)
        ax.set_title("Tempo de entrega (dias) por Projeto", fontsize=12)
        # Configurar as labels do eixo x
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias_unicas, fontsize=8, rotation=15)  # Ajuste o tamanho e a rotação aqui

        # Exibir o gráfico no aplicativo Streamlit
        c5.pyplot(fig)
                    # Definir as categorias (projetos) e os valores (Desempenho %)
        categorias = df_display_normalizado_copy["Projeto"]
        valores = df_display_normalizado_copy["Tempo de entrega (dias)"]



        # Obter categorias únicas
        categorias_unicas = categorias.unique()

        # Número de categorias únicas
        num_categorias = len(categorias_unicas)

        # Calcular os ângulos para cada categoria única no gráfico de radar
        angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
        angulos += angulos[:1]

        # Mapear valores de desempenho para os ângulos correspondentes
        valores_mapeados = [valores[categorias == categoria].mean() for categoria in categorias_unicas]

        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})

        # Plotar os valores mapeados para cada categoria única
        ax.fill(angulos, valores_mapeados + valores_mapeados[:1], 'b',color='green', alpha=0.3)
        ax.set_title("Retorno sobre o investimento (ROI) por Projeto", fontsize=12)
        # Configurar as labels do eixo x
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias_unicas, fontsize=8, rotation=15)  # Ajuste o tamanho e a rotação aqui

        # Exibir o gráfico no aplicativo Streamlit
        c6.pyplot(fig)
        ##########################################################################################################3

        # Definir as categorias (projetos) e os valores (Desempenho %)
        categorias = df_display_normalizado_copy["Projeto"]
        valores = df_display_normalizado_copy["Sustentabilidade (nota de 0 a 10)"]



        # Obter categorias únicas
        categorias_unicas = categorias.unique()

        # Número de categorias únicas
        num_categorias = len(categorias_unicas)

        # Calcular os ângulos para cada categoria única no gráfico de radar
        angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
        angulos += angulos[:1]

        # Mapear valores de desempenho para os ângulos correspondentes
        valores_mapeados = [valores[categorias == categoria].mean() for categoria in categorias_unicas]

        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})

        # Plotar os valores mapeados para cada categoria única
        ax.fill(angulos, valores_mapeados + valores_mapeados[:1], 'b',color='red', alpha=0.3)
        ax.set_title("Sustentabilidade (nota de 0 a 10) por Projeto", fontsize=12)
        # Configurar as labels do eixo x
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias_unicas, fontsize=8, rotation=15)  # Ajuste o tamanho e a rotação aqui

        # Exibir o gráfico no aplicativo Streamlit
        c7.pyplot(fig)
                    # Definir as categorias (projetos) e os valores (Desempenho %)
        categorias = df_display_normalizado_copy["Projeto"]
        valores = df_display_normalizado_copy["Nível de automação (escala de 0 a 5)"]



        # Obter categorias únicas
        categorias_unicas = categorias.unique()

        # Número de categorias únicas
        num_categorias = len(categorias_unicas)

        # Calcular os ângulos para cada categoria única no gráfico de radar
        angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
        angulos += angulos[:1]

        # Mapear valores de desempenho para os ângulos correspondentes
        valores_mapeados = [valores[categorias == categoria].mean() for categoria in categorias_unicas]

        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})

        # Plotar os valores mapeados para cada categoria única
        ax.fill(angulos, valores_mapeados + valores_mapeados[:1], 'b',color='purple', alpha=0.3)
        ax.set_title("Nível de automação (escala de 0 a 5) por Projeto", fontsize=12)
        # Configurar as labels do eixo x
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias_unicas, fontsize=8, rotation=15)  # Ajuste o tamanho e a rotação aqui

        # Exibir o gráfico no aplicativo Streamlit
        c8.pyplot(fig)
        #######################################################################################################################


        # Definir as categorias (projetos) e os valores (Desempenho %)
        categorias = df_display_normalizado_copy["Projeto"]
        valores = df_display_normalizado_copy["Redução de emissões de CO2 (g/km)"]



        # Obter categorias únicas
        categorias_unicas = categorias.unique()

        # Número de categorias únicas
        num_categorias = len(categorias_unicas)

        # Calcular os ângulos para cada categoria única no gráfico de radar
        angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
        angulos += angulos[:1]

        # Mapear valores de desempenho para os ângulos correspondentes
        valores_mapeados = [valores[categorias == categoria].mean() for categoria in categorias_unicas]

        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})

        # Plotar os valores mapeados para cada categoria única
        ax.fill(angulos, valores_mapeados + valores_mapeados[:1], 'b',color='brown', alpha=0.3)
        ax.set_title("Redução de emissões de CO2 (g/km) por Projeto", fontsize=12)
        # Configurar as labels do eixo x
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias_unicas, fontsize=8, rotation=15)  # Ajuste o tamanho e a rotação aqui

        # Exibir o gráfico no aplicativo Streamlit
        c9.pyplot(fig)
                    # Definir as categorias (projetos) e os valores (Desempenho %)
        categorias = df_display_normalizado_copy["Projeto"]
        valores = df_display_normalizado_copy["Risco (nota de 0 a 10)"]



        # Obter categorias únicas
        categorias_unicas = categorias.unique()

        # Número de categorias únicas
        num_categorias = len(categorias_unicas)

        # Calcular os ângulos para cada categoria única no gráfico de radar
        angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
        angulos += angulos[:1]

        # Mapear valores de desempenho para os ângulos correspondentes
        valores_mapeados = [valores[categorias == categoria].mean() for categoria in categorias_unicas]

        # Plotar o gráfico de radar
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})

        # Plotar os valores mapeados para cada categoria única
        ax.fill(angulos, valores_mapeados + valores_mapeados[:1], 'b',color='blue', alpha=0.3)
        ax.set_title("Risco (nota de 0 a 10) por Projeto", fontsize=12)
        # Configurar as labels do eixo x
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias_unicas, fontsize=8, rotation=15)  # Ajuste o tamanho e a rotação aqui

        # Exibir o gráfico no aplicativo Streamlit
        c10.pyplot(fig)
        st.title("Conclusão")
        # Texto com tamanho de fonte personalizado
        texto = """
        Após a aplicação do Método AHP Gaussiano para avaliar o desempenho dos projetos com base nos critérios relevantes, chegamos a conclusões valiosas. Este método nos permitiu quantificar e comparar objetivamente o desempenho de cada projeto, considerando a importância relativa de cada critério na tomada de decisão. No entanto, reconhecemos que os pesos atribuídos a cada critério podem variar de acordo com as preferências do decisor. Se o decisor desejar personalizar a ponderação dos critérios de acordo com suas prioridades específicas, oferecemos a opção de aplicar o Método ProPPAga. O Método ProPPAga é uma abordagem flexível que permite ao decisor ajustar os pesos dos critérios de acordo com suas preferências individuais. Para utilizar o Método ProPPAga, basta selecionar a opção correspondente nas opções do nosso aplicativo a esquerda. Isso permitirá ao decisor refinar ainda mais a avaliação dos projetos, personalizando os pesos dos critérios de acordo com suas necessidades e objetivos específicos. Dessa forma, garantimos que as decisões tomadas sejam ainda mais alinhadas com as preferências e metas do decisor, contribuindo para uma tomada de decisão informada e estratégica.
        """

        # Exibir o texto com tamanho de fonte personalizado
        st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>{texto}</p>", unsafe_allow_html=True)
        # Espaço entre os métodos
        st.markdown('---')
        pass
elif paginaSelecionada == 'Otimização de Processo': 
    # Função para resolver o problema de programação linear inteira
    def solve_distribution(values, total):
        # Criar um problema de maximização
        prob = pulp.LpProblem("ServiceDistribution", pulp.LpMaximize)


        # Variáveis: quantidade de cada serviço
        quantities = {service: pulp.LpVariable(f"q_{service}", 1, cat=pulp.LpInteger) for service in values}

        # Função objetivo: aproximar-se do valor total sem excedê-lo
        prob += pulp.lpSum([quantities[service] * val for service, val in values.items()])

        # Restrição: o total não deve exceder o valor desejado
        prob += pulp.lpSum([quantities[service] * val for service, val in values.items()]) <= total

        # Restrição adicional: minimizar a diferença entre as quantidades
        for s1 in values:
            for s2 in values:
                if s1 != s2:
                    prob += quantities[s1] - quantities[s2] >= -1
                    prob += quantities[s1] - quantities[s2] <= 1

        # Resolver o problema
        prob.solve()

        # Verificar se há uma solução
        if prob.status != pulp.LpStatusOptimal:
            return "Não foi possível encontrar uma solução ótima."

        # Retornar os resultados
        return {service: int(quantities[service].varValue) for service in values}

    # Interface do usuário no Streamlit
    def main():
        st.title("Distribuição de Serviços")

        # Entrada do usuário para valores dos serviços e valor total
        st.sidebar.title("Valores dos Serviços")
        num_services = st.sidebar.number_input("Quantos serviços diferentes?", min_value=2, value=3)
        values = {}
        for i in range(num_services):
            service_name = st.sidebar.text_input(f"Nome do Serviço {i+1}", f"Serviço {i+1}")
            service_value = st.sidebar.number_input(f"Valor do {service_name}", min_value=0.01)
            values[service_name] = service_value

        total_value = st.sidebar.number_input("Valor Total Desejado", min_value=0.01)

        # Botão para calcular distribuição
        if st.sidebar.button("Calcular Distribuição"):
            result = solve_distribution(values, total_value)

            # Criar um DataFrame para exibir os resultados
            df = pd.DataFrame(columns=["Quantidade", "Valor Unitário", "Total por Serviço"])
            for service, quantity in result.items():
                df.loc[service] = [quantity, values[service], quantity * values[service]]

            # Adicionar linha com somas e diferença
            total_distribution = df["Total por Serviço"].sum()
            df.loc["Total"] = ["-", "-", total_distribution]
            df.loc["Diferença"] = ["-", "-", total_value - total_distribution]

            # Exibir o DataFrame
            st.write("Resultado da Distribuição:")
            st.dataframe(df)

    if __name__ == "__main__":
        main()             
elif paginaSelecionada == 'Método ProPPAGA':
   
            st.title("Análise Multicritério de Projetos - Método ProPPAGA")
            #st.image('propaga.png', caption='', use_column_width=True)
            col1l, col2l = st.columns(2)
            tamanho_fonte = 24
            # Coluna da esquerda com a imagem
            largura_coluna = 700
            largura_imagem = int(largura_coluna * 0.8)  # 60% da largura da coluna
            imagem = 'propaga4.png'
            # Coluna da esquerda com a imagem
           
                
            with col1l:
                st.image(imagem, caption='', width=largura_imagem)

            # Coluna da direita com o texto
            with col2l:
                st.markdown("""
                <div style='font-size: 24px; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 60vh;'>
                O método ProPPAga, desenvolvido pelo Instituto Militar de Engenharia (IME), baseia-se na premissa de que as alternativas se comportam de forma Gaussiana dentro de cada critério. Isso implica que, ao analisar um conjunto de alternativas, espera-se que elas exibam um comportamento médio e que se distribuam normalmente em torno dessa média. É importante destacar que, para a aplicação do método ProPPAga, não é necessário confirmar se essa presunção é de fato verdadeira. Em outras palavras, não é preciso realizar nenhum teste de aderência para verificar se as alternativas realmente seguem uma distribuição Gaussiana.
                </div>
                """, unsafe_allow_html=True)
    # Título da aplicação
    
            st.table(df)
            st.image('fluxo.png', caption='', use_column_width=True)

         

            
            # Carregue o DataFrame com os dados do arquivo
            

            # Lista de critérios
            criterios = df.columns[1:]

            # Número de critérios
            num_crit = len(criterios)

            # Defina o valor S_jmax com base no número de critérios
            S_jmax = 7 if num_crit <=7 else num_crit

            # Crie controles de seleção para os critérios
            st.markdown(f"<p style='font-size:24px;'>Escolha os critérios nos quais o decisor deseja ponderar seu peso:</p>", unsafe_allow_html=True)
            crit_selected = st.multiselect("", criterios, key="select_crit")

            # Aplicar o tamanho da fonte de 24px para o texto
            


            if crit_selected:
                # Matriz de decisão com os critérios selecionados
                X = df[crit_selected].values

                # Pesos dos critérios (inicializados com 1)
                pesos = [1.0] * len(crit_selected)

                # Identifique os critérios de custo e aplique a multiplicação por -1
                st.markdown(f"<p style='font-size:{tamanho_fonte}px;'>Primeiro, o decisor escolhe os critérios monotônicos, ou seja, os critérios em que um valor menor é considerado melhor:</p>", unsafe_allow_html=True)
                custo_selected = st.multiselect("", crit_selected)
                for crit in custo_selected:
                    df[crit] = df[crit] * (-1)

                # Defina o grau de importância (S_j) para cada critério usando sliders
                st.subheader("Atribua um grau de importância  para cada critério:")
                graus_importancia = {}
                for crit in crit_selected:
                    grau_importancia = st.slider(f"Grau de Importância para {crit}:", 1, S_jmax, 1)
                    graus_importancia[crit] = grau_importancia

                # Calcula o somatório dos graus de importância
                soma_graus_importancia = sum(graus_importancia.values())
                # Calcula o somatório dos graus de importância
                soma_graus_importancia = sum(graus_importancia.values())

                # Calcula os pesos para cada critério
                for crit in crit_selected:
                    pesos[crit_selected.index(crit)] = graus_importancia[crit] / soma_graus_importancia

                # Crie uma nova coluna no DataFrame com as pontuações normalizadas
                pontuacao = calcular_pontuacao(X, pesos)
                df['Pontuação'] = pontuacao

                # Calcula os pesos para cada critério
                for crit in crit_selected:
                    pesos[crit_selected.index(crit)] = graus_importancia[crit] / soma_graus_importancia

                # Calcule a média e o desvio padrão para cada critério
                medias = df[crit_selected].mean()
                desvios_padrao = df[crit_selected].std()

                # Crie um DataFrame com as médias e desvios padrão
                estatisticas_crit = pd.DataFrame({'Média': medias, 'Desvio Padrão': desvios_padrao})

                # Exiba o DataFrame com as estatísticas
                st.subheader("Média e Desvio Padrão para Cada Critério:")
                st.table(estatisticas_crit)

                # Exiba o DataFrame original junto com a coluna de pontuação
                st.subheader("DataFrame Original com Modificação, Pontuação e Estatísticas:")
                st.table(df)

                # Calcule o desempenho normalizado para cada critério
                calcular_desempenho_normalizado(df, crit_selected)

                # Exiba o DataFrame com o desempenho normalizado
                st.subheader("Tabela  Desempenho:")
        # Salvar a primeira coluna em uma variável temporária
                primeira_coluna = df.iloc[:, 0]

                # Excluir a primeira coluna temporariamente para realizar os cálculos
                df = df.iloc[:, 1:]

                # Função para calcular a soma dos valores na linha após a multiplicação pela 'Pontuação'
                colunas_normalizadas = [col for col in df.columns if "Normalizado" in col]

                # Função para calcular a soma dos valores nas colunas normalizadas após a multiplicação pela 'Pontuação'
                def calcular_soma(row):
                    return sum(row[col] * row['Pontuação'] for col in colunas_normalizadas)


                # Aplicar a função a cada linha do DataFrame e armazenar os resultados em uma nova coluna 'Soma'
                df['Soma'] = df.apply(calcular_soma, axis=1)

                # Restaurar a primeira coluna no DataFrame
                df.insert(0, 'Projeto', primeira_coluna)
                soma_total = df['Soma'].sum()

                # Função para normalizar a coluna 'Soma'
                def normalizar_soma(valor):
                    return valor / soma_total

                # Aplicar a função de normalização à coluna 'Soma'
                df['Desempenho Geral'] = df['Soma'].apply(normalizar_soma)
                df = df.sort_values(by='Desempenho Geral', ascending=False)

    # Calcular a porcentagem com duas casas decimais
                maior_valor = df['Desempenho Geral'].max()

    # Normalize os valores
                df['Desempenho Geral %'] = (df['Desempenho Geral'] / maior_valor * 100).round(2).astype(str) + '%'

                # Selecionar a primeira coluna e as 4 últimas colunas
                colunas_selecionadas = df.iloc[:, [0] + list(range(-4, 0))]

                # Exibir o DataFrame colunas_selecionadas
                st.table(colunas_selecionadas)
                # Crie um DataFrame com os projetos e a coluna 'Desempenho Geral %'
                data = df[['Projeto', 'Desempenho Geral %']]

                # Crie o gráfico de radar
                fig = px.line_polar(data, r='Desempenho Geral %', theta='Projeto', line_close=True)

                # Atualize o layout do gráfico
                fig.update_layout(polar=dict(radialaxis=dict(showticklabels=False, gridcolor='gray')))
                fig.update_layout(title='Análise dos Projetos ordenados')
                # Exiba o gráfico no Streamlit
                st.plotly_chart(fig)
            

elif paginaSelecionada == 'Simulação de Entrega':
        st.title("Entrega  de Projetos - Método Monte Carlo")
        df = pd.read_excel("projetos.xlsx")
        df.rename(columns={'nome_projeto': 'Projeto'}, inplace=True)
        col2x,col1x = st.columns(2)
        largura_coluna = 700
        largura_imagem = int(largura_coluna * 0.8)  # 60% da largura da coluna
        imagem = 'mcnp.png'

        # Coluna da esquerda com a imagem
        with col1x:
            st.image(imagem, caption='', width=largura_imagem)
        # Coluna da direita com o texto
        with col2x:
            st.markdown("""
            <div style='font-size: 24px; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 70vh;'>
           O Método de Monte Carlo, originado no Projeto Manhattan para resolver problemas de física nuclear, como o transporte de nêutrons, foi desenvolvido por John von Neumann, Stanislaw Ulam e Nicholas Metropolis. Inicialmente usado para simular o comportamento aleatório dos nêutrons, sua aplicação evoluiu para a gestão de projetos, onde hoje é uma ferramenta valiosa para calcular a probabilidade de diferentes resultados em projetos. Na gestão de projetos, o método auxilia na avaliação de riscos e no planejamento, simulando cenários variados com base na incerteza e aleatoriedade. Isso possibilita uma melhor compreensão dos potenciais riscos e prazos, preparando gerentes de projeto para imprevistos. A imagem conceitual ilustra essa transição do método da ciência pura para a gestão de projetos, simbolizando a união da física nuclear com estratégias modernas de gerenciamento.
            </div>
            """, unsafe_allow_html=True)
        
         
        st.subheader("Avaliação de Riscos e Planejamento com Simulação Estatística")
        col2x2,col1x2 = st.columns(2)
        #largura_coluna = 700
       # largura_imagem = int(largura_coluna * 0.8)  # 60% da largura da coluna
        #imagem = 'mcnp.png'

        # Coluna da esquerda com a imagem
        with col1x2:
            st.table(df.head(8))
        # Coluna da direita com o texto
        with col2x2:
            st.markdown("""
            <div style='font-size: 24px; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh;'>
           O Método de Monte Carlo é uma ferramenta crucial no gerenciamento de riscos de entrega de projetos, particularmente na indústria automotiva. Utilizando dados históricos que incluem montadoras, tipos de projeto, localizações, e datas de início e término, este método simula diversos cenários para prever a probabilidade de cumprir datas de entrega propostas. Ao analisar projetos semelhantes, ele oferece uma estimativa baseada em probabilidades, apoiando os gerentes de projeto na tomada de decisões informadas sobre prazos. Essencialmente, o Método de Monte Carlo transforma dados históricos em insights práticos para otimizar a gestão e entrega de projetos.
            </div>
            """, unsafe_allow_html=True)
        
        simulacoes_acima_referencia = df2[df2["Marco"] > df2["Proposta"]]
        mediaSimula = simulacoes_acima_referencia["Marco"].mean()
        desvioPadraoSimula = simulacoes_acima_referencia["Marco"].std()

        # Seletores para tipos de projeto, montadora e localização
        tipo_projeto_escolhido = st.selectbox("Escolha o tipo de projeto", df['tipo_projeto'].unique())
        montadora_escolhida = st.selectbox("Escolha a montadora", df['montadora'].unique())
        localizacao_escolhida = st.selectbox("Escolha a localização", df['localizacao'].unique())

        df_filtrado = df[(df['tipo_projeto'] == tipo_projeto_escolhido) & 
                        (df['montadora'] == montadora_escolhida) & 
                        (df['localizacao'] == localizacao_escolhida)]

        # Calcular a média e o desvio padrão da duração
        media_duracao = df_filtrado['duracao'].mean()
        
        desvio_padrao_duracao = df_filtrado['duracao'].std()
        # Exibir a média e o desvio padrão
        st.write(f"Média da duração: {round(media_duracao,2)} dias")
        st.write(f"Desvio padrão da duração: {round(desvio_padrao_duracao,2)} dias")
        # Campos para entrada de dados do novo projeto
        nome_projeto = st.text_input("Nome do Projeto em estudo")
        #data_inicio = st.date_input("Data de Início")
        #data_final = st.date_input("Data Final")
        # Campo de entrada numérica
        valor_numerico = st.slider("Selecione uma Duração Proposta para o Projeto", min_value=650, max_value=3500, value=1000)

        # Exibe o valor escolhido
        st.write(f"Valor numérico escolhido: {valor_numerico}")

        # Botão para calcular a diferença em dias
        #if st.button("Calcular Duração"):
            #if data_inicio and data_final:
           
                #st.write(f"A duração do projeto '{nome_projeto}' é de {round(duracao,2)} dias.")
           # else:
                #st.error("Por favor, insira datas de início e final válidas.")
        duracao = valor_numerico
                

        if st.button("Simular"):
            
            st.write(f"Tempo restante: {duracao}")
            media_atividades=duracao
            desvio_projeto=desvio_padrao_duracao
        

 
            media_projeto=media_duracao

            desvio_atividades = 1  # Define o desvio padrão das atividades
            data_atual = datetime.datetime.now().date()
            probabilidade_entrega, resultados_projeto, resultados_atividades = simular_monte_carlo(
                media_projeto, desvio_projeto, media_atividades, desvio_atividades
            )
            
            st.subheader("Probabilidade de Entrega:")
            Probabilidadel=1 - probabilidade_entrega
            st.write(f"{(1 - probabilidade_entrega) * 100:.2f}%")
            # Exiba as datas na interface Streamlit
            #st.write(f"Data Provável: {data_probavel_formatada}")
            #st.write(f"Data Pessimista: {data_pessimista_formatada}")
            #st.write(f"Data Otimista: {simulacoes_acima_referencia}")
            # Após calcular 'mediaAtraso' e 'Sd', calcule as datas desejadas

            x = resultados_atividades
            y = resultados_projeto
            # Filtrar o DataFrame com base nas escolhas
            # Define a condição para simulações acima da proposta
            condicao_vermelho = resultados_projeto > media_atividades

            data_atual = datetime.datetime.now().date()

            # Calcula a Data Provável
            somavermelho = np.sum(resultados_projeto > media_atividades)
            if somavermelho >= 3:
                media_valores_vermelhos = np.mean(resultados_projeto[resultados_projeto > media_atividades])
            else:
                media_valores_vermelhos = np.mean(resultados_projeto[resultados_projeto <= media_atividades])

            sd_valores_vermelhos = np.std(resultados_projeto[resultados_projeto > media_atividades])
            dias_adicionais = int(media_valores_vermelhos) + int(sd_valores_vermelhos)
            data_probavel = data_atual + timedelta(days=dias_adicionais)
            data_probavel_formatada = data_probavel.strftime("%d/%m/%Y")

            # Calcula a Probabilidade de Entrega
            probabilidade_entrega_formatada = f"Probabilidade de Entrega(Prazo inicial): {Probabilidadel * 100:.2f}%"
            # Cria a figura e o eixo para o gráfico de dispersão
            fig, ax1 = plt.subplots(figsize=(12, 7))

            # Plota os pontos verdes e vermelhos
            verdes = ax1.scatter(resultados_projeto[~(resultados_projeto > media_atividades)], resultados_atividades[~(resultados_projeto > media_atividades)], c='green', marker='o', s=5, alpha=0.15,label='Dentro do Prazo')
            vermelhos = ax1.scatter(resultados_projeto[resultados_projeto > media_atividades], resultados_atividades[resultados_projeto > media_atividades], c='red', marker='o', s=5, alpha=0.05, label='Fora do Prazo')

            # Linha de Proposta de Prazo
            line1 = ax1.axvline(x=media_atividades, color='blue', linestyle='--', label='Proposta de Prazo')
            ax1.set_xlabel('Simulação Temporal de Projetos')
            ax1.set_ylabel('Tempo do Projeto em estudo')

            # Cria o segundo eixo para a curva de frequência acumulada
            ax2 = ax1.twinx()

            frequencia_acumulada_inicial = (1 - probabilidade_entrega) * 100
            frequencias_acumuladas = [frequencia_acumulada_inicial]
            dias_acumulados = [media_atividades]
            datas_acumuladas = [data_atual + datetime.timedelta(days=media_atividades)]

            dias = media_atividades
            limite_maximo = 98
            while frequencias_acumuladas[-1] < limite_maximo:
                dias += 90
                dias_acumulados.append(dias)
                datas_acumuladas.append(data_atual + datetime.timedelta(days=dias))
                proporcao_atual = min(limite_maximo, np.sum(resultados_projeto <= dias) / len(resultados_projeto) * 100)
                frequencias_acumuladas.append(proporcao_atual)

            # Plota segmentos de reta, pontos e anotações para a frequência acumulada
            for i in range(len(dias_acumulados) - 1):  # Exclui o último ponto
                line2, = ax2.plot(dias_acumulados[i:i+2], frequencias_acumuladas[i:i+2], 'm-', label='Probabilidade Acumulada' if i == 0 else "")
                ax2.scatter(dias_acumulados[i], frequencias_acumuladas[i], color='blue', s=50)
                label = "Prazo inicial\n" if i == 0 else ""
                xytext = (20, -18) if i == len(dias_acumulados) - 2 else (20, -18)
                ax2.annotate(label + f'{frequencias_acumuladas[i]:.2f}%\n{datas_acumuladas[i].strftime("%d/%m/%Y")}',
                            (dias_acumulados[i], frequencias_acumuladas[i]),
                            textcoords="offset points",
                            xytext=xytext,
                            ha='left')

            ax2.set_ylabel('Probabilidade Acumulada(%)', color='m')
            somavermelho=np.count_nonzero(condicao_vermelho)
            somanaoverde = np.count_nonzero(~condicao_vermelho)
            legend_elements = [
            line1, 
            verdes, 
            vermelhos, 
            plt.Line2D([0], [0], color='m', lw=2),
            plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='blue', markersize=10, label=data_probavel_formatada),
            plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='green', markersize=10, label=probabilidade_entrega_formatada)
            ]
            labels = [
                'Proposta de Prazo ' + str(media_atividades) + ' dias', 
                'Simulações Dentro do Prazo :'+ str(somanaoverde), 
                'Simulações Fora do Prazo :'+ str(somavermelho), 
                'Probabilidade Acumulada', 
                'Data Provável Sugerida: ' + data_probavel_formatada, 
                probabilidade_entrega_formatada
            ]
            plt.legend(handles=legend_elements, labels=labels, loc='upper left')

            plt.title(f'Simulação de Monte Carlo com Probabilidade Acumulada - {nome_projeto}')
            plt.tight_layout()
            st.pyplot(fig)
elif paginaSelecionada == 'Programação Inteira': 
    st.title("Análise Multicritério de Projetos - Programação Linear Inteira ")
    col2x2t,col1x2t = st.columns(2)
        #largura_coluna = 700
       # largura_imagem = int(largura_coluna * 0.8)  # 60% da largura da coluna
        #imagem = 'mcnp.png'
    largura_coluna = 700
    largura_imagem = int(largura_coluna * 0.8)  # 60% da largura da coluna
    imagem = 'pi.png'
    # Coluna da esquerda com a imagem
    with col1x2t:
        st.image(imagem, caption='', width=largura_imagem)
    # Coluna da direita com o texto
    with col2x2t:
        st.markdown("""
        <div style='font-size: 24px; line-height: 1.5; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 60vh;'>
        A programação inteira é uma técnica de otimização matemática crucial na gestão de custo-benefício. Essencial para problemas onde variáveis são inteiras (como na alocação de recursos), ela oferece soluções exatas, respeitando restrições orçamentárias e maximizando a eficiência. Esta abordagem permite decisões baseadas em dados, otimizando o retorno sobre o investimento e minimizando custos, tornando-se uma ferramenta valiosa para planejamento estratégico e tomada de decisões. Além disso, a programação inteira permite incorporar restrições rígidas nos modelos, assegurando que todas as soluções atendam a critérios específicos. Isso é especialmente útil na gestão de custo-benefício, onde as restrições orçamentárias e de recursos desempenham um papel fundamental. A capacidade de modelar essas restrições de forma precisa leva a decisões mais informadas e eficazes, maximizando o retorno sobre o investimento e minimizando os custos desnecessários.
        </div>
        """, unsafe_allow_html=True)


 
    import locale

    # Define a localização para o formato de moeda (por exemplo, o Brasil)
    locale.setlocale(locale.LC_ALL, '')


    st.title("Seleção de Projetos")

    from pulp import LpProblem, LpVariable, lpSum, value
   
    # Ler o arquivo Excel e exibir os dados
   
    df.rename(columns={'nome_projeto': 'Projeto'}, inplace=True)
    st.write("Dados do arquivo Excel:")
    st.table(df)
        # Definir os valores mínimo e máximo para o controle deslizante com base na coluna "Custo"
    valor_minimo = df['Custo de transporte (R$)'].min()
    valor_maximo = df['Custo de transporte (R$)'].sum()

    # Criar o controle deslizante com valores reais
    valor_maximo_investimento = st.slider(
        "Digite o valor máximo de investimento:",
        min_value=float(valor_minimo),
        max_value=float(valor_maximo),
        step=0.01,
        format="%.2f"
    )
    valor_maximo_investimento_formatado = f"{valor_maximo_investimento:,.2f}"

    st.text(f"Valor máximo de investimento: R$:{valor_maximo_investimento_formatado}")

    
    # Definir as variáveis de decisão para cada projeto como binárias (0 ou 1)
    projetos = df['Projeto']
    prob = LpProblem("Selecao_de_Projetos", LpMaximize)
    x = LpVariable.dict("Projeto_Selecionado", projetos, cat=LpBinary)

    # Definir a função objetivo para maximizar o lucro total
    prob += lpSum(df.loc[i, 'Lucro Esperado'] * x[projeto] for i, projeto in enumerate(projetos))

    # Adicionar a restrição de valor máximo
    #valor_maximo = st.number_input("Digite o valor máximo de investimento:", format="%.2f")
    prob += lpSum(df.loc[i, 'Custo de transporte (R$)'] * x[projeto] for i, projeto in enumerate(projetos)) <= valor_maximo_investimento
    # Resolver o modelo
    prob.solve()

    # Exibir os resultados
    st.write("\nResultados:")
    for projeto in projetos:
        if x[projeto].varValue == 1:
            st.write(f"Projeto Selecionado: {projeto}")
            lucro_total = value(prob.objective)
            lucro_total_formatado = f"{lucro_total:,.2f}"
    st.write(f"Lucro Total: R$:{lucro_total_formatado}", font_size=18)
    projetos_selecionados = [projeto for projeto in projetos if x[projeto].varValue == 1]

    # Filtrar o DataFrame original com os projetos selecionados
    df_selecionados = df[df['Projeto'].isin(projetos_selecionados)]
    # Filtrar o DataFrame original para incluir apenas os projetos selecionados
    df_selecionados = df[df['Projeto'].isin(projetos_selecionados)].copy()
    

    # Calcular a coluna 'Custo_beneficio'
    df_selecionados['Custo_beneficio'] = df_selecionados['Custo de transporte (R$)'] / df_selecionados['Lucro Esperado']
    st.table(df_selecionados)
    # Selecionar apenas as colunas 'Projeto' e 'Custo_beneficio' de df_selecionados para o gráfico
    data = df_selecionados[['Projeto', 'Custo_beneficio']]

    # Crie o gráfico de radar
    fig = px.line_polar(data, r='Custo_beneficio', theta='Projeto', line_close=True)
    fig.update_layout(title='Análise de Custo-Benefício dos Projetos Selecionados')
    # Exibir o gráfico
    st.plotly_chart(fig)
