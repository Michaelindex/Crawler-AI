# To run this code you need to install the following dependencies:
# pip install google-genai
# pip install pandas

import base64
import os
from google import genai
from google.genai import types
import json
import re
import time
import pandas as pd
from datetime import datetime

def extract_json_from_text(text):
    """Extrai o JSON da resposta do modelo."""
    try:
        # Procura por um bloco JSON na resposta
        json_match = re.search(r'```json\s*({[\s\S]*?})\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"Erro ao extrair JSON: {e}")
        return {}

def build_prompt(current_data, existing_data):
    """Constrói o prompt com os dados acumulados e existentes."""
    known_data = "Dados existentes do médico:\n"
    for key, value in existing_data.items():
        if value and pd.notna(value):  # Só inclui campos não vazios
            known_data += f"  {key}: \"{value}\"\n"
    
    known_data += "\nDados encontrados até agora:\n"
    for key, value in current_data.items():
        if value and key not in existing_data:  # Só inclui novos dados encontrados
            known_data += f"  {key}: \"{value}\"\n"
    
    return f"""Você é um assistente que pesquisa dados de médicos. 
{known_data}
                                
Campos faltantes a buscar de cada médico:
  - Endereço completo (logradouro, número, complemento, bairro, CEP, cidade, estado)
  - Telefone fixo do local de trabalho (até 2 números)
  - Celular (até 2 números)
  - E-mail (até 2 endereços)
**Formato de resposta:** JSON com as chaves  
{{"especialidade_medica","endereco_completo_a1","numero_a1","complemento_a1","bairro_a1","cep_a1","cidade_a1","estado_a1","telefone1","telefone2","celular1","celular2","email1","email2"}}.
## Exemplo de Resposta Esperada (JSON)  
```json
{{
  "especialidade_medica": "Cardiologia",
  "endereco_completo_a1": "Rua das Flores, 123, Sala 45, Centro, São Paulo, SP, 01000-000",
  "numero_a1": "123",
  "complemento_a1": "Sala 45",
  "bairro_a1": "Centro",
  "cep_a1": "01000000",
  "cidade_a1": "São Paulo",
  "estado_a1": "SP",
  "telefone1": "(11) 1234-5678",
  "telefone2": "",
  "celular1": "(11) 98765-4321",
  "celular2": "",
  "email1": "fulano.tal@hospital.org",
  "email2": ""
}}```"""

def process_doctor(client, model, doctor_data):
    """Processa um único médico com 6 iterações."""
    start_time = time.time()  # Início do processamento do médico
    
    # Dados existentes do médico
    existing_data = {
        "Nome": f"{doctor_data['Nome']} {doctor_data['Sobrenome']}".strip(),
        "CRM": str(doctor_data['CRM']),
        "UF": str(doctor_data['UF']),
        "Status_CRM": str(doctor_data['STATUS_CRM']) if pd.notna(doctor_data['STATUS_CRM']) else '',
        "Especialidade": str(doctor_data['Especialidade Médica']) if pd.notna(doctor_data['Especialidade Médica']) else '',
        "Endereco": str(doctor_data['Endereco Completo']) if pd.notna(doctor_data['Endereco Completo']) else '',
        "Logradouro": str(doctor_data['Logradouro']) if pd.notna(doctor_data['Logradouro']) else '',
        "Numero": str(doctor_data['Numero']) if pd.notna(doctor_data['Numero']) else '',
        "Complemento": str(doctor_data['Complemento']) if pd.notna(doctor_data['Complemento']) else '',
        "Bairro": str(doctor_data['Bairro']) if pd.notna(doctor_data['Bairro']) else '',
        "CEP": str(doctor_data['CEP']) if pd.notna(doctor_data['CEP']) else '',
        "Cidade": str(doctor_data['Cidade']) if pd.notna(doctor_data['Cidade']) else '',
        "Estado": str(doctor_data['Estado']) if pd.notna(doctor_data['Estado']) else '',
        "Telefone1": str(doctor_data['Telefone A1']) if pd.notna(doctor_data['Telefone A1']) else '',
        "Telefone2": str(doctor_data['Telefone A2']) if pd.notna(doctor_data['Telefone A2']) else '',
        "Celular1": str(doctor_data['Celular A1']) if pd.notna(doctor_data['Celular A1']) else '',
        "Celular2": str(doctor_data['Celular A2']) if pd.notna(doctor_data['Celular A2']) else '',
        "Email1": str(doctor_data['E-mail A1']) if pd.notna(doctor_data['E-mail A1']) else '',
        "Email2": str(doctor_data['E-mail A2']) if pd.notna(doctor_data['E-mail A2']) else ''
    }
    
    # Dados iniciais para busca
    current_data = {
        "especialidade_medica": existing_data['Especialidade'],
        "endereco_completo_a1": existing_data['Endereco'],
        "numero_a1": existing_data['Numero'],
        "complemento_a1": existing_data['Complemento'],
        "bairro_a1": existing_data['Bairro'],
        "cep_a1": existing_data['CEP'],
        "cidade_a1": existing_data['Cidade'],
        "estado_a1": existing_data['Estado'],
        "telefone1": existing_data['Telefone1'],
        "telefone2": existing_data['Telefone2'],
        "celular1": existing_data['Celular1'],
        "celular2": existing_data['Celular2'],
        "email1": existing_data['Email1'],
        "email2": existing_data['Email2']
    }

    print(f"\nDados existentes do médico:")
    print(json.dumps(existing_data, indent=2, ensure_ascii=False))

    # Loop de 6 iterações
    for iteration in range(6):
        iteration_start_time = time.time()  # Início da iteração
        print(f"\n=== Iteração {iteration + 1} ===")
        
        # Delay incremental de 7 segundos, com 45 segundos na última iteração
        if iteration > 0:
            if iteration == 5:  # Última iteração
                delay = 45
            else:
                delay = 7 * iteration
            print(f"Aguardando {delay} segundos antes da próxima requisição...")
            time.sleep(delay)
        
        # Construir o prompt com os dados atuais e existentes
        prompt_text = build_prompt(current_data, existing_data)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
                ],
            ),
        ]
        
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            tools=tools,
            response_mime_type="text/plain",
        )

        max_retries = 3  # Número máximo de tentativas
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Coletar a resposta completa
                full_response = ""
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    print(chunk.text, end="")
                    full_response += chunk.text

                # Extrair e atualizar dados
                new_data = extract_json_from_text(full_response)
                for key, value in new_data.items():
                    if value and isinstance(value, str) and (not current_data.get(key) or len(value) > len(current_data.get(key, ""))):
                        current_data[key] = value
                        
                iteration_time = time.time() - iteration_start_time
                print(f"\nTempo da iteração {iteration + 1}: {iteration_time:.2f} segundos")
                break  # Se chegou aqui, deu tudo certo, então sai do loop de retry
                
            except Exception as e:
                retry_count += 1
                print(f"\nErro durante a iteração {iteration + 1} (tentativa {retry_count} de {max_retries}): {e}")
                
                if retry_count < max_retries:
                    print("Aguardando 30 segundos antes de tentar novamente...")
                    time.sleep(30)
                else:
                    print(f"Número máximo de tentativas atingido para a iteração {iteration + 1}")
                    break

    total_time = time.time() - start_time
    current_data['tempo_total_processamento'] = f"{total_time:.2f} segundos"
    print(f"\nTempo total de processamento do médico: {total_time:.2f} segundos")
    
    return current_data

def generate():
    start_time_total = time.time()  # Início do processamento total
    start_datetime = datetime.now()  # Data e hora de início
    
    # Lendo a chave da API do arquivo
    with open('../../apis/gemini.key', 'r') as file:
        api_key = file.read().strip()

    client = genai.Client(
        api_key=api_key
    )

    model = "gemini-2.5-flash-preview-04-17"
    
    # Lendo o arquivo CSV
    df = pd.read_csv('input.csv')
    print(f"\nTotal de médicos encontrados no CSV: {len(df)}")
    
    # Lista para armazenar os resultados
    results = []
    
    # Processando cada médico
    total_doctors = len(df)
    for index, row in df.iterrows():
        print(f"\n\n=== Processando médico {index + 1} de {total_doctors} ===")
        print(f"CRM: {row['CRM']} - Nome: {row['Nome']} {row['Sobrenome']}")
        
        try:
            result = process_doctor(client, model, row)
            results.append({
                "crm": row['CRM'],
                "nome": f"{row['Nome']} {row['Sobrenome']}".strip(),
                "dados": result
            })
            
            # Pequeno delay entre médicos
            if index < total_doctors - 1:
                print("\nAguardando 5 segundos antes do próximo médico...")
                time.sleep(5)
                
        except Exception as e:
            print(f"Erro ao processar médico {row['CRM']}: {e}")
            results.append({
                "crm": row['CRM'],
                "nome": f"{row['Nome']} {row['Sobrenome']}".strip(),
                "erro": str(e)
            })
    
    # Salvando os resultados
    end_datetime = datetime.now()  # Data e hora de fim
    total_time = time.time() - start_time_total
    output_file = f'output_{start_datetime.strftime("%Y%m%d_%H%M%S")}.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Informações de Processamento ===\n")
        f.write(f"Início: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Fim: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Tempo Total de Processamento: {total_time:.2f} segundos\n\n")
        
        for result in results:
            f.write(f"\n=== Médico: {result['crm']} - {result['nome']} ===\n")
            if 'erro' in result:
                f.write(f"ERRO: {result['erro']}\n")
            else:
                f.write(json.dumps(result['dados'], indent=2, ensure_ascii=False))
            f.write("\n\n")
    
    print(f"\nProcessamento concluído! Resultados salvos em {output_file}")
    print(f"Total de médicos processados: {len(results)}")
    print(f"Início: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Fim: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Tempo total de processamento: {total_time:.2f} segundos")

if __name__ == "__main__":
    generate() 