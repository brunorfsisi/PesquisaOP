import streamlit as st
##Godigo teste extrair imagem
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
def processar_texto_bloco(texto):
    partes = texto.split('\n')  # Dividir o texto em linhas

    # Este exemplo supõe que cada parte do texto corresponde diretamente a uma coluna.
    # Ajuste os índices conforme a estrutura do seu texto.
    return {
        "Ida": partes[0],
        "Tipo1": partes[1],
        "Tipo": partes[2],
        "Tipo2": partes[3],
        "Data_ida": partes[4],
        "Hora": partes[5],
        "Aeroporto": partes[6],
        "Loc_Partida": partes[7],
        "Hora_chegada": partes[8],
        "Aeroporto_destino": partes[9],
        "Localidade_destino": partes[10],
        "Duracao": partes[11],
        "Bagagem": partes[11],
        "Volta": partes[12],
        "Tipo_volta1": partes[13],
        "Tipo_volta": partes[14],
        "Tipo_volta2": partes[15],
        "Data_volta": partes[16],
        "Hora_volta_partida": partes[17],
        "Aeroporto_p_volta": partes[18],
        "Localidade_partida_volta": partes[19],
        "Hora_chegada_volta": partes[20],
        "Aeroporto_chegada_volta": partes[21],
        "Localidade_chegada_volta": partes[22],
        "Duracao_volta": partes[21],
        "Bagagem_volta": partes[22],
        "Pontos": partes[23],
        "Desconto": partes[24],
        "Pontos_com_desc": partes[25],
        "Clube": partes[26],
        "Valor_por_adulto": partes[27],
        "Valor_por_adulto2": partes[28]
    }
st.text_input
# URL da página da web
url = "https://livelo.com.br/passagens-aereas/trip/BHZ/MIA/2024-01-30/2024-02-27/ADULT/ECONOMY_CLASS/INTERNATIONAL"  # Substitua com a URL real da página

# Configurações do ChromeDriver
chrome_options = ChromeOptions()
#chrome_options.add_argument("--headless")  # Executa o Chrome em modo headless (sem interface gráfica)
driver = webdriver.Chrome(options=chrome_options)
