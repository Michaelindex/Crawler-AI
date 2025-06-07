import pandas as pd
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
import concurrent.futures
import time

def load_api_keys():
    """Carrega as chaves da API dos arquivos."""
    keys = []
    for i in range(1, 5):
        key_file = f"../../../apis/gemini{i}.key" if i > 1 else "../../../apis/gemini.key"
        with open(key_file, 'r') as f:
            keys.append(f.read().strip())
    return keys

def build_prompt(row):
    """Constrói o prompt para padronização dos dados."""
    return f"""Analise e padronize os seguintes dados de um médico:

Dados atuais:
{json.dumps(row.to_dict(), indent=2, ensure_ascii=False)}

Regras de padronização:
1. Especialidade: Apenas o nome da especialidade, sem explicações ou comentários
2. Endereço: Separar o endereço completo em suas partes:
   - Logradouro: Nome da rua/avenida
   - Número: Apenas o número
   - Complemento: Sala, andar, etc.
   - Bairro: Nome do bairro
   - CEP: Formato 00000-000
   - Cidade: Nome da cidade
   - Estado: UF
3. Telefones e Celulares: Padronizar no formato +55(DDD)XXXXX-XXXX
   - Remover números incompletos ou com X
   - Adicionar +55 se não existir
   - Formatar com parênteses e hífen
4. E-mails: Manter em minúsculas

Retorne um JSON com os dados padronizados, mantendo a mesma estrutura do input mas com os dados corrigidos."""

def process_row(row, api_key):
    """Processa uma linha usando a API do Gemini."""
    client = genai.Client(api_key=api_key)
    model = "gemini-1.5-flash"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=build_prompt(row)),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="text/plain",
    )
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extrair JSON da resposta
        response_text = response.text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            print(f"Erro ao extrair JSON da resposta para CRM {row['CRM']}")
            return row.to_dict()
            
    except Exception as e:
        print(f"Erro ao processar CRM {row['CRM']}: {str(e)}")
        return row.to_dict()

def process_chunk(chunk, api_key):
    """Processa um chunk de linhas usando uma chave da API."""
    results = []
    for _, row in chunk.iterrows():
        result = process_row(row, api_key)
        results.append(result)
        time.sleep(1)  # Pequeno delay para evitar rate limits
    return results

def main():
    # Carregar chaves da API
    api_keys = load_api_keys()
    
    # Ler o CSV
    df = pd.read_csv('output_20250605_004549.csv')
    
    # Dividir o DataFrame em chunks para processamento paralelo
    chunk_size = len(df) // len(api_keys)
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Processar chunks em paralelo
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk, api_keys[i]): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_results = future.result()
            all_results.extend(chunk_results)
    
    # Criar novo DataFrame com resultados
    result_df = pd.DataFrame(all_results)
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'output_standardized_{timestamp}.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Arquivo padronizado salvo como: {output_file}")

if __name__ == "__main__":
    main() 