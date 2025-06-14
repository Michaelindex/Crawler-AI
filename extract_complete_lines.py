import pandas as pd
import numpy as np

def count_filled_fields(row):
    """Conta quantos campos estão preenchidos em uma linha"""
    return row.count()

def extract_complete_lines():
    # Lê o arquivo CSV
    df = pd.read_csv('output.csv')
    
    # Define as cidades e quantidades necessárias
    cities = {
        'São Paulo': 30,
        'Rio de Janeiro': 240,
        'Belo Horizonte': 60,
        'Curitiba': 50
    }
    
    # Lista para armazenar as linhas selecionadas
    selected_rows = []
    
    # Para cada cidade
    for city, quantity in cities.items():
        # Filtra as linhas da cidade
        city_df = df[df['City A1'] == city].copy()
        
        # Conta campos preenchidos em cada linha
        city_df['filled_fields'] = city_df.apply(count_filled_fields, axis=1)
        
        # Ordena por quantidade de campos preenchidos (decrescente)
        city_df = city_df.sort_values('filled_fields', ascending=False)
        
        # Seleciona a quantidade necessária
        selected = city_df.head(quantity)
        
        # Adiciona à lista de selecionados
        selected_rows.append(selected)
    
    # Combina todas as linhas selecionadas
    final_df = pd.concat(selected_rows)
    
    # Remove a coluna auxiliar de contagem
    final_df = final_df.drop('filled_fields', axis=1)
    
    # Salva o resultado
    final_df.to_csv('output-extract.csv', index=False)
    
    # Imprime estatísticas
    print("\nEstatísticas de extração:")
    for city in cities:
        count = len(final_df[final_df['City A1'] == city])
        print(f"{city}: {count} linhas extraídas")

if __name__ == "__main__":
    extract_complete_lines() 