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
        "Nome": doctor_data['Name'],
        "CRM": doctor_data['CRM'],
        "UF": doctor_data['UF'],
        "Status_CRM": doctor_data['STATUS_CRM'],
        "Especialidade": doctor_data['Specialty'],
        "Endereco": doctor_data['Address'],
        "Contato": doctor_data['Contact_Information']
    }
    
    # Dados iniciais para busca
    current_data = {
        "especialidade_medica": existing_data['Especialidade'],
        "endereco_completo_a1": existing_data['Endereco'],
        "numero_a1": "",
        "complemento_a1": "",
        "bairro_a1": "",
        "cep_a1": "",
        "cidade_a1": "",
        "estado_a1": "",
        "telefone1": "",
        "telefone2": "",
        "celular1": "",
        "celular2": "",
        "email1": "",
        "email2": ""
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
    
    # Preparar dados para o CSV
    output_data = {
        'CRM': doctor_data['CRM'],
        'UF': doctor_data['UF'],
        'STATUS_CRM': doctor_data['STATUS_CRM'],
        'Name': doctor_data['Name'],
        'Specialty': current_data.get('especialidade_medica', ''),
        'Address': current_data.get('endereco_completo_a1', ''),
        'Contact_Information': '; '.join(filter(None, [
            current_data.get('telefone1', ''),
            current_data.get('telefone2', ''),
            current_data.get('celular1', ''),
            current_data.get('celular2', ''),
            current_data.get('email1', ''),
            current_data.get('email2', '')
        ])),
        'Tempo_Processamento': current_data.get('tempo_total_processamento', '')
    }
    
    return output_data

def generate():
    start_time_total = time.time()  # Início do processamento total
    start_datetime = datetime.now()  # Data e hora de início
    
    # Lendo a chave da API do arquivo
    with open('../../../apis/gemini.key', 'r') as file:
        api_key = file.read().strip()

    client = genai.Client(
        api_key=api_key
    )

    model = "gemini-2.5-flash-preview-04-17"
    
    # Lendo o arquivo CSV
    df = pd.read_csv('../input.csv')
    print(f"\nTotal de médicos encontrados no CSV: {len(df)}")
    
    # Criando DataFrame para output
    output_df = df.copy()
    output_df['Tempo_Processamento'] = None
    
    # Lista para armazenar os resultados
    results = []
    
    # Processando cada médico
    for index, row in df.iterrows():
        print(f"\nProcessando médico {index + 1} de {len(df)}")
        
        # Função auxiliar para converter valores para string de forma segura
        def safe_str(value):
            if pd.isna(value):
                return ''
            return str(value).strip()
        
        # Preparando dados iniciais
        current_data = {
            'CRM': safe_str(row['CRM']),
            'UF': safe_str(row['UF']),
            'STATUS_CRM': safe_str(row['STATUS_CRM']),
            'Name': f"{safe_str(row['Nome'])} {safe_str(row['Sobrenome'])}".strip(),
            'Specialty': safe_str(row['Especialidade Médica']),
            'Address': safe_str(row['Endereco Completo']),
            'Contact_Information': '; '.join(filter(None, [
                safe_str(row['Telefone A1']),
                safe_str(row['Telefone A2']),
                safe_str(row['Celular A1']),
                safe_str(row['Celular A2']),
                safe_str(row['E-mail A1']),
                safe_str(row['E-mail A2'])
            ]))
        }
        
        print(f"\nDados existentes do médico:")
        for key, value in current_data.items():
            if value and not pd.isna(value):
                print(f"- {key}: {value}")
        
        try:
            # Processando o médico
            result = process_doctor(client, model, current_data)
            results.append(result)
            
            # Atualizando o DataFrame de output
            output_df.at[index, 'Nome'] = result['Name'].split()[0] if result['Name'] else ''
            output_df.at[index, 'Sobrenome'] = ' '.join(result['Name'].split()[1:]) if result['Name'] else ''
            output_df.at[index, 'Especialidade Médica'] = result['Specialty']
            output_df.at[index, 'Endereco Completo'] = result['Address']
            
            # Atualizando campos de contato
            if result['Contact_Information']:
                contacts = result['Contact_Information'].split(';')
                for i, contact in enumerate(contacts):
                    contact = contact.strip()
                    if '@' in contact:  # É um email
                        if i == 0:
                            output_df.at[index, 'E-mail A1'] = contact
                        else:
                            output_df.at[index, 'E-mail A2'] = contact
                    elif contact.startswith('('):  # É um telefone fixo
                        if i == 0:
                            output_df.at[index, 'Telefone A1'] = contact
                        else:
                            output_df.at[index, 'Telefone A2'] = contact
                    else:  # É um celular
                        if i == 0:
                            output_df.at[index, 'Celular A1'] = contact
                        else:
                            output_df.at[index, 'Celular A2'] = contact
            
            output_df.at[index, 'Tempo_Processamento'] = result['Tempo_Processamento']
            
        except Exception as e:
            print(f"Erro ao processar médico {current_data['Name']}: {str(e)}")
            output_df.at[index, 'Tempo_Processamento'] = "ERRO"
            # Mantendo a mesma estrutura do result em caso de erro
            results.append({
                'CRM': current_data['CRM'],
                'UF': current_data['UF'],
                'STATUS_CRM': current_data['STATUS_CRM'],
                'Name': current_data['Name'],
                'Specialty': current_data['Specialty'],
                'Address': current_data['Address'],
                'Contact_Information': current_data['Contact_Information'],
                'Tempo_Processamento': "ERRO",
                'Erro': str(e)
            })
    
    # Salvando os resultados
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    
    # Salvando CSV
    output_csv = f'output_{timestamp}.csv'
    output_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Salvando log
    output_log = f'output_{timestamp}.txt'
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write(f"Início do processamento: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Fim do processamento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Tempo total de processamento: {time.time() - start_time_total:.2f} segundos\n\n")
        
        for result in results:
            f.write(f"Médico: {result['Name']}\n")
            f.write(f"CRM: {result['CRM']}\n")
            f.write(f"UF: {result['UF']}\n")
            f.write(f"Status: {result['STATUS_CRM']}\n")
            f.write(f"Especialidade: {result['Specialty']}\n")
            f.write(f"Endereço: {result['Address']}\n")
            f.write(f"Contato: {result['Contact_Information']}\n")
            f.write(f"Tempo de processamento: {result['Tempo_Processamento']}\n")
            if 'Erro' in result:
                f.write(f"Erro: {result['Erro']}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"\nProcessamento concluído!")
    print(f"Resultados salvos em:")
    print(f"- CSV: {output_csv}")
    print(f"- Log: {output_log}")
    print(f"Tempo total de processamento: {time.time() - start_time_total:.2f} segundos")

if __name__ == "__main__":
    generate() 